//! Small multilayer perceptron evaluator ("distilled" from the perfect database).
//!
//! Input features (72, mover's point of view; see docs/FORMATS.md):
//!   [0, 29)   mover occupancy by compact square index
//!   [29, 58)  opponent occupancy by compact square index
//!   [58, 64)  one-hot number of mover pieces borne off (0..=5)
//!   [64, 70)  one-hot number of opponent pieces borne off (0..=5)
//!   70        mover remaining distance  sum(31 - s) / 100
//!   71        opponent remaining distance / 100
//! Hidden layers use ReLU; the output is a logit (sigmoid gives P(mover wins)).
//!
//! File format (little-endian; docs/FORMATS.md): b"SNN1", u32 layer count, then per layer
//! u32 in, u32 out, f32 weights[out][in], f32 bias[out].

use crate::board::{Bits, MAX_PIECES, OFF, Pos};
use crate::eval::Evaluator;
use crate::index::{N_USABLE, compact};
use crate::invalid_data;
use std::io;
use std::path::Path;

const MY_SQUARES: usize = 0;
const OPP_SQUARES: usize = MY_SQUARES + N_USABLE as usize;
const MY_OFF: usize = OPP_SQUARES + N_USABLE as usize;
const OPP_OFF: usize = MY_OFF + MAX_PIECES + 1;
const MY_DISTANCE: usize = OPP_OFF + MAX_PIECES + 1;
const OPP_DISTANCE: usize = MY_DISTANCE + 1;
/// Number of input features.
pub const N_INPUTS: usize = OPP_DISTANCE + 1;
const _: () = assert!(N_INPUTS == 72, "docs/FORMATS.md and the Python and JS ports use 72 features");

/// Largest layer count and layer width of an SNN1 network (docs/FORMATS.md; a guard
/// against corrupt headers, far above any network this project trains).
pub const MAX_LAYERS: usize = 16;
pub const MAX_WIDTH: usize = 4096;

/// The largest feature value: a remaining distance, at most (30 + 29 + 28 + 27 + 26) / 100.
/// Occupancies and one-hots are 0 or 1.
const MAX_FEATURE: f64 = 1.4;

/// The non-zero features of a position as `(index, value)` pairs.
fn sparse_features(pos: Pos) -> impl Iterator<Item = (usize, f32)> {
    let squares = |mask: u32, base: usize| Bits(compact(mask)).map(move |i| (base + i as usize, 1.0));
    let borne_off = |mask: u32| MAX_PIECES.saturating_sub(mask.count_ones() as usize);
    let distance = |mask: u32| Bits(mask).map(|s| OFF - s).sum::<u32>() as f32 / 100.0;
    squares(pos.me, MY_SQUARES).chain(squares(pos.opp, OPP_SQUARES)).chain([
        (MY_OFF + borne_off(pos.me), 1.0),
        (OPP_OFF + borne_off(pos.opp), 1.0),
        (MY_DISTANCE, distance(pos.me)),
        (OPP_DISTANCE, distance(pos.opp)),
    ])
}

/// The feature vector of a position (as the Python and JS ports compute it).
pub fn dense_features(pos: Pos) -> [f32; N_INPUTS] {
    let mut v = [0f32; N_INPUTS];
    for (i, x) in sparse_features(pos) {
        v[i] = x;
    }
    v
}

/// A fully connected layer.
struct Dense {
    n_in: usize,
    n_out: usize,
    /// Weights, `[out][in]` as in the file until `Net::from_bytes` stores them `[in][out]`,
    /// so that each non-zero input adds one contiguous column.
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Dense {
    /// Adds `x` times the weights of input `i` to `out`.
    #[inline(always)]
    fn add_input(&self, out: &mut [f32], i: usize, x: f32) {
        for (o, &w) in out.iter_mut().zip(&self.w[i * self.n_out..][..self.n_out]) {
            *o += x * w;
        }
    }
}

pub struct Net {
    layers: Vec<Dense>,
    /// The largest layer output count.
    width: usize,
}

/// Reads the little-endian fields of a network file.
struct Fields<'a>(&'a [u8]);

impl Fields<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.0.len() < n {
            return Err(invalid_data("truncated network file"));
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    /// A `u32` count or size.
    fn count(&mut self) -> io::Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize)
    }

    fn f32s(&mut self, n: usize) -> io::Result<Vec<f32>> {
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
    }
}

impl Net {
    pub fn load(path: &Path) -> io::Result<Net> {
        Net::from_bytes(&std::fs::read(path)?)
    }

    /// Parses an SNN1 network, checking that its shape is 72 -> ... -> 1 within the
    /// format's limits, that its parameters are finite, that no activation can overflow
    /// `f32` (so that every logit is finite) and that no bytes are left over.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Net> {
        let mut f = Fields(bytes);
        if f.take(4)? != b"SNN1" {
            return Err(invalid_data("not an SNN1 network file"));
        }
        let n_layers = f.count()?;
        if !(1..=MAX_LAYERS).contains(&n_layers) {
            return Err(invalid_data(format!("{n_layers} layers; a network has 1..={MAX_LAYERS}")));
        }
        let mut layers = Vec::with_capacity(n_layers);
        let mut n_in_expected = N_INPUTS;
        for k in 0..n_layers {
            let (n_in, n_out) = (f.count()?, f.count()?);
            if n_in != n_in_expected || !(1..=MAX_WIDTH).contains(&n_out) {
                return Err(invalid_data(format!(
                    "layer {k} is {n_in} -> {n_out}; expected {n_in_expected} inputs and 1..={MAX_WIDTH} outputs"
                )));
            }
            let w = f.f32s(n_in * n_out)?;
            let b = f.f32s(n_out)?;
            if !w.iter().chain(&b).all(|x| x.is_finite()) {
                return Err(invalid_data(format!("layer {k} has a parameter that is not finite")));
            }
            layers.push(Dense { n_in, n_out, w, b });
            n_in_expected = n_out;
        }
        if n_in_expected != 1 {
            return Err(invalid_data("the network must have one output"));
        }
        if !f.0.is_empty() {
            return Err(invalid_data("trailing bytes after the network"));
        }
        // Bounds on the absolute values of each layer's inputs: the features, then the
        // previous layer's outputs. Sums of terms within half the range of f32 cannot
        // overflow it, however they are rounded.
        let mut bound = vec![MAX_FEATURE; N_INPUTS];
        for (k, layer) in layers.iter().enumerate() {
            bound = (layer.b.iter().zip(layer.w.chunks_exact(layer.n_in)))
                .map(|(b, row)| b.abs() as f64 + row.iter().zip(&bound).map(|(w, x)| w.abs() as f64 * x).sum::<f64>())
                .collect();
            let max = bound.iter().copied().fold(0.0, f64::max);
            if max > f32::MAX as f64 / 2.0 {
                return Err(invalid_data(format!("layer {k}'s outputs could overflow f32 (up to {max:.1e})")));
            }
        }
        for layer in &mut layers {
            let (n_in, n_out) = (layer.n_in, layer.n_out);
            layer.w = (0..n_in * n_out).map(|k| layer.w[(k % n_out) * n_in + k / n_out]).collect();
        }
        let width = layers.iter().map(|layer| layer.n_out).max().expect("at least one layer");
        Ok(Net { layers, width })
    }

    /// Raw output logit.
    pub fn logit(&self, pos: Pos) -> f32 {
        // Buffers of this call's own: the network is shared between threads.
        let mut buffers = vec![0f32; 2 * self.width];
        let (mut h, mut next) = buffers.split_at_mut(self.width);
        let first = &self.layers[0];
        h[..first.n_out].copy_from_slice(&first.b);
        for (i, x) in sparse_features(pos) {
            first.add_input(&mut h[..first.n_out], i, x);
        }
        for layer in &self.layers[1..] {
            let out = &mut next[..layer.n_out];
            out.copy_from_slice(&layer.b);
            // ReLU: an input at or below zero adds nothing.
            for (i, &x) in h[..layer.n_in].iter().enumerate() {
                if x > 0.0 {
                    layer.add_input(out, i, x);
                }
            }
            std::mem::swap(&mut h, &mut next);
        }
        h[0]
    }
}

impl Evaluator for Net {
    fn value(&self, pos: Pos) -> f64 {
        pos.terminal_value().unwrap_or_else(|| 1.0 / (1.0 + (-(self.logit(pos) as f64)).exp()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Rules;

    /// SNN1 bytes for a network with the given `(n_in, n_out, weights, biases)` layers.
    fn snn1(layers: &[(usize, usize, Vec<f32>, Vec<f32>)]) -> Vec<u8> {
        let mut out = b"SNN1".to_vec();
        out.extend((layers.len() as u32).to_le_bytes());
        for (n_in, n_out, w, b) in layers {
            out.extend((*n_in as u32).to_le_bytes());
            out.extend((*n_out as u32).to_le_bytes());
            out.extend(w.iter().chain(b).flat_map(|x| x.to_le_bytes()));
        }
        out
    }

    /// The message of the error that loading `bytes` fails with.
    fn load_error(bytes: &[u8]) -> String {
        match Net::from_bytes(bytes) {
            Ok(_) => panic!("the network loaded"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn features_of_the_start_position() {
        let f = dense_features(Rules::KENDALL5.start());
        assert_eq!(f[..10], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(f[OPP_SQUARES + 1], 1.0);
        assert_eq!((f[MY_OFF], f[OPP_OFF]), (1.0, 1.0));
        assert_eq!(f[MY_DISTANCE], (30 + 28 + 26 + 24 + 22) as f32 / 100.0);
        assert_eq!(f.iter().filter(|&&x| x != 0.0).count(), 14);
    }

    #[test]
    fn evaluates_a_two_layer_network() {
        // Hidden unit 0 = mover distance, unit 1 = -opponent distance + 0.5; output =
        // 2 * relu(h0) + 3 * relu(h1) - 1.
        let mut w1 = vec![0f32; 2 * N_INPUTS];
        w1[MY_DISTANCE] = 1.0;
        w1[N_INPUTS + OPP_DISTANCE] = -1.0;
        let bytes = snn1(&[(N_INPUTS, 2, w1, vec![0.0, 0.5]), (2, 1, vec![2.0, 3.0], vec![-1.0])]);
        let net = Net::from_bytes(&bytes).unwrap();
        let pos = Pos::from_squares(&[21], &[30]).unwrap();
        assert!((net.logit(pos) - (2.0 * 0.10 + 3.0 * 0.49 - 1.0)).abs() < 1e-6);

        for bad in [&bytes[..bytes.len() - 1], &[bytes.as_slice(), &[0]].concat(), &bytes[..3]] {
            assert!(Net::from_bytes(bad).is_err());
        }
        let wrong_inputs = snn1(&[(N_INPUTS - 1, 1, vec![0.0; N_INPUTS - 1], vec![0.0])]);
        assert!(Net::from_bytes(&wrong_inputs).is_err());
        let two_outputs = snn1(&[(N_INPUTS, 2, vec![0.0; 2 * N_INPUTS], vec![0.0; 2])]);
        assert!(Net::from_bytes(&two_outputs).is_err());
        let too_wide = snn1(&[
            (N_INPUTS, MAX_WIDTH + 1, vec![0.0; (MAX_WIDTH + 1) * N_INPUTS], vec![0.0; MAX_WIDTH + 1]),
            (MAX_WIDTH + 1, 1, vec![0.0; MAX_WIDTH + 1], vec![0.0]),
        ]);
        assert!(load_error(&too_wide).contains("1..=4096 outputs"));
        for x in [f32::NAN, f32::INFINITY] {
            let mut w = vec![0f32; N_INPUTS];
            w[7] = x;
            let err = load_error(&snn1(&[(N_INPUTS, 1, w, vec![0.0])]));
            assert!(err.contains("not finite"), "{err}");
        }
    }

    #[test]
    fn a_network_that_could_overflow_is_refused() {
        // Two hidden units of 3e38 times the mover's distance, and their difference: the
        // units overflow to infinity and the output is not a number.
        let mut w1 = vec![0f32; 2 * N_INPUTS];
        (w1[MY_DISTANCE], w1[N_INPUTS + MY_DISTANCE]) = (3e38, 3e38);
        let hidden = |w1: Vec<f32>| snn1(&[(N_INPUTS, 2, w1, vec![0.0; 2]), (2, 1, vec![1.0, -1.0], vec![0.0])]);
        let err = load_error(&hidden(w1.clone()));
        assert!(err.contains("layer 0's outputs could overflow"), "{err}");
        // Weights that only overflow when summed in the next layer.
        (w1[MY_DISTANCE], w1[N_INPUTS + MY_DISTANCE]) = (1e19, 1e19);
        let bytes = snn1(&[(N_INPUTS, 2, w1.clone(), vec![0.0; 2]), (2, 1, vec![2e19, 2e19], vec![0.0])]);
        let err = load_error(&bytes);
        assert!(err.contains("layer 1's outputs could overflow"), "{err}");
        // Large weights within range load, and give a finite logit.
        let net = Net::from_bytes(&hidden(w1)).unwrap();
        assert_eq!(net.logit(Rules::KENDALL5.start()), 0.0);
    }

    /// The offsets of the u32 header fields of a valid SNN1 file: the layer count, then
    /// each layer's input and output counts.
    fn header_fields(layers: &[(usize, usize, Vec<f32>, Vec<f32>)]) -> Vec<usize> {
        let mut fields = vec![4];
        let mut offset = 8;
        for (n_in, n_out, _, _) in layers {
            fields.extend([offset, offset + 4]);
            offset += 8 + 4 * (n_in * n_out + n_out);
        }
        fields
    }

    #[test]
    fn mutated_files_are_refused_or_load_without_panicking() {
        use crate::rng::Rng;
        const FIELDS: [u32; 12] = [0, 1, 2, 3, 16, 17, 71, 72, 73, 4096, 4097, u32::MAX];
        const FLOATS: [f32; 6] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, -0.0, 1e-45];
        let bases: Vec<_> = [
            vec![(N_INPUTS, 1, vec![0.25; N_INPUTS], vec![0.5])],
            vec![(N_INPUTS, 3, vec![0.1; 3 * N_INPUTS], vec![0.0; 3]), (3, 1, vec![0.5, -0.5, 1.0], vec![0.0])],
        ]
        .into_iter()
        .map(|layers| (snn1(&layers), header_fields(&layers)))
        .collect();
        let start = Rules::KENDALL5.start();
        let mut rng = Rng::new(3);
        let (mut loaded, mut refused) = (0, 0);
        for _ in 0..5000 {
            let (base, fields) = &bases[rng.below(bases.len() as u64) as usize];
            let mut bytes = base.clone();
            let mut pick = |n: usize| rng.below(n as u64) as usize;
            let len = bytes.len();
            // A header field set to a boundary value, a word to a special float, a flipped
            // bit, or bytes cut off, added or cut out.
            match pick(6) {
                0 => {
                    let at = fields[pick(fields.len())];
                    bytes[at..at + 4].copy_from_slice(&FIELDS[pick(FIELDS.len())].to_le_bytes());
                }
                1 => {
                    let at = 4 * pick(len / 4);
                    bytes[at..at + 4].copy_from_slice(&FLOATS[pick(FLOATS.len())].to_le_bytes());
                }
                2 => {
                    let at = pick(len);
                    bytes[at] ^= 1 << pick(8);
                }
                3 => bytes.truncate(pick(len)),
                4 => {
                    let n = 1 + pick(8);
                    bytes.extend((0..n).map(|_| pick(256) as u8));
                }
                _ => {
                    let at = pick(len);
                    let end = (at + 1 + pick(8)).min(len);
                    drop(bytes.drain(at..end));
                }
            }
            match Net::from_bytes(&bytes) {
                Ok(net) => {
                    assert!(net.logit(start).is_finite());
                    loaded += 1;
                }
                Err(_) => refused += 1,
            }
        }
        assert!(loaded > 500 && refused > 500, "{loaded} loaded, {refused} refused");
    }

    #[test]
    fn loads_the_committed_network() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/senet_net.bin");
        let net = Net::load(&path).unwrap();
        let v = net.value(Rules::KENDALL5.start());
        assert!((v - 0.5025).abs() < 0.01, "V(start) = {v}");
    }
}
