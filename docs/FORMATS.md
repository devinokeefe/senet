# Data formats and indexing

## Perfect-play database (`db/kendall5/`)

The solver computes the value of **every** position of the 5-piece game:
`V(me, opp)` = probability that the player about to throw wins under optimal play
by both sides. Positions are grouped into *layers* by piece counts.

### Squares and compact indices

Square 27 (House of Water) is never occupied, so only 29 squares can hold a piece:

```
usable squares U = [1, 2, ..., 26, 28, 29, 30]
compact index c(s) = s - 1   for s <= 26
                     s - 2   for s in {28, 29, 30}
```

### Layer `(w, b)`

* `w` = number of the mover's pieces still on the board (1..5)
* `b` = number of the opponent's pieces still on the board (1..5)
* (A side with 0 pieces on the board has already won, so such positions are terminal
  and are not stored.)
* Layer size `N(w, b) = C(29, w) * C(29 - w, b)`.

### Index of a position within its layer

1. Let `m_0 < m_1 < ... < m_{w-1}` be the compact indices of the mover's squares.
   `me_rank = sum_i C(m_i, i + 1)` (colexicographic rank; `C(n, k) = 0` when `n < k`).
2. For each opponent compact index `o`, let `o' = o - |{ i : m_i < o }|`
   (its position among the `29 - w` compact squares the mover does not occupy).
   Sort them `o'_0 < ... < o'_{b-1}` and let `opp_rank = sum_j C(o'_j, j + 1)`.
3. `index = me_rank * C(29 - w, b) + opp_rank`.

### Files

* `db/kendall5/L{w}{b}.f32` — raw little-endian `float32` array of length `N(w, b)`;
  entry `index` holds `V(me, opp)`.
* `db/kendall5/meta.json` — `ruleset` (`"kendall5"`), `layer_sizes` (`"L{w}{b}"` →
  `N(w, b)`), `groups` and, once the 5v5 layer is solved, `start_value_white`. `groups`
  holds a record per solved group, keyed by its layers as in `"[(5, 4), (4, 5)]"`: its
  `layers`, `states`, `sweeps`, `final_max_delta` (the largest change of the last sweep),
  `seconds` and `tol` (the tolerance the solve stopped at). A statistic that was not
  recorded is `null`. A `note` says what a record leaves out or where it comes from: a
  solve that resumed from a checkpoint counts only its last run's sweeps and seconds, and
  a group that a rerun finds solved without a record has no statistics. `senet check`
  reports how many records have notes or no statistics.
* `db/kendall5/MANIFEST.json` — the database's [integrity manifest](#integrity-manifests).
  `senet solve` removes it before changing the database and writes it once the database
  is complete. A solve that finds every layer already solved changes nothing and leaves
  the manifest as it is.

A partial database (such as a solve in progress leaves) is accepted if it holds, with each
layer `(w, b)`, the layers its moves lead to: `(b, w)`, `(w - 1, b)` and `(b, w - 1)`. Every
position it covers can then be valued move by move from the database alone.

While solving, the directory also holds working files, all little-endian:

* `solve.lock` — present while a solve uses the directory, so that a second solve stops
  with a message. A solve that is killed leaves it behind: delete it by hand.
* `L{w}{b}.f32.ckpt` — checkpoint of a layer being solved in `float32` (same layout as
  `L{w}{b}.f32`).
* `L{w}{b}.u24.ckpt` — checkpoint of a layer being solved in 24-bit fixed point: 3 bytes
  per position, value = `q / (2^24 - 1)` (the 5v5 layer is solved this way).
* `L{w}{b}.u24` — 24-bit copy of a solved layer, the memory-mapped input of a 24-bit
  group. Deleted when the group is finished.
* `*.tmp` — a file being written; it is renamed into place once complete.

A rerun of `senet solve` skips the groups whose `.f32` files exist and resumes the others
from their checkpoints, after removing the working files that an interrupted solve left
and no longer needs. A checkpoint is overwritten in place, and deleted before its layer's
final file is written, which keeps the peak disk use of a full solve at 36.8 GB for the
34.2 GB database. A checkpoint cut short by a crash is therefore a mix of two sweeps'
values, which the iteration continues from all the same.

## Move-generation dump (JSON Lines)

Produced by `senet dump-moves` and consumed by the independent Python checker
(`python -m senet_ref.check_movegen`) and the browser-engine checker
(`tools/check_js_rules.mjs`). One JSON object per line (in any key order):

```json
{"me": [3, 9, 26], "opp": [4, 5, 28], "t": 2,
 "moves": [
   {"from": 9, "to": 11, "kind": "move", "dir": "fwd", "me": [3, 11, 26], "opp": [4, 5, 28]},
   {"from": 26, "to": 28, "kind": "swap", "dir": "fwd", "me": [3, 9, 28], "opp": [4, 5, 26]}
 ]}
```

* `me` / `opp`: sorted lists of occupied squares before the move (mover's view), each
  with 1 to 5 pieces: a game in progress.
* `t`: throw value 1..5.
* `moves`: every legal move for that throw, sorted by (`from`, `to`).
  * `to` is the square where the piece **finally rests**: 31 when borne off; for a
    House of Water move it is the square the piece is sent back to (15 or lower).
  * `kind`: `"move"` (to an empty square), `"swap"`, `"off"` (borne off), `"water"`.
  * `dir`: `"fwd"` or `"back"`.
  * `me` / `opp`: sorted occupied squares after the move (still the same mover's view).
* An empty `moves` list means the turn is forfeited.

All numbers are JSON integers; the checkers reject a dump with no records, and records
with other values, rather than converting them.

## Training records (`senet gen-data`)

A flat array of 12-byte little-endian records: `u32` mover mask, `u32` opponent mask
(bit `i` = square `i`, mover's view) and `f32` value `V(me, opp)` from the database. Each
record is a game in progress (1 to 5 pieces per side, on distinct squares other than 27)
with a value in [0, 1]; `senet_train.distill` refuses a file with any other record or with
a partial one at its end.

`gen-data` writes the file whole (through `NAME.tmp`, renamed into place) and then its
[manifest](#integrity-manifests) `NAME.manifest.json`, which records the settings
(`games`, `eps`, `uniform`, `seed`), the database (by the SHA-256 of its manifest) and
the number of records. It refuses an output whose manifest would replace that of another
file (`train.dat` beside `train.bin`, whose manifest is also `train.manifest.json`). `senet_train.distill` verifies the manifest, hashing the file,
before it reads a record.

## Network file (SNN1)

A multilayer perceptron with ReLU hidden layers whose single output is a logit
(sigmoid = probability that the player about to throw wins). Little-endian:

```
b"SNN1"   u32 layer count
per layer: u32 n_in, u32 n_out, f32 weights[n_out][n_in], f32 bias[n_out]
```

A network has 1 to 16 layers with 1 to 4096 outputs each (limits that keep a corrupt
header from asking for gigabytes, far above any network this project trains). The first
layer has 72 inputs, each layer's `n_in` is the previous layer's `n_out`, the last layer
has one output, every weight and bias is finite, and the file ends after the last layer.
No activation may be able to overflow `float32`: with inputs of at most 1.4, the bound
`|bias| + sum(|weight| * input bound)` must stay within half of `float32`'s range at every
unit of every layer. Rust (`senet_core::net`), Python (`senet_train.snn1`, which the
training code and the portable build use) and the browser engine
(`web/portable/senet-engine.js`) reject anything else.

`senet_train.distill --out NAME` writes the network `NAME.bin` and its PyTorch state dict
`NAME.pt`, each whole, and then their [manifest](#integrity-manifests)
`NAME.manifest.json`. The manifest records the settings, the training data (by the SHA-256
of its manifest), and the software and device (`created_by.environment`). Its `contents`
hold:

| Key | Value |
|---|---|
| `architecture`, `parameters` | the layer widths and the number of parameters |
| `features` | the input encoding: its name (`senet-features/1`) and layout |
| `split` | the validation split: `val_frac`, `seed`, the NumPy version, the data's SHA-256, the numbers of records and of distinct positions, and its definition |
| `errors` | the mean absolute, root-mean-square and largest error on the `validation records` and on those at `unseen positions` (positions in no training record) |
| `epoch_seconds` | how long each epoch took (a network trained by this version) |

`distill --evaluate NET` measures a network's errors again. If the network has a manifest,
it uses that manifest's split, and refuses other training data or settings unless given
`--force`.

### Input features (72, mover's point of view)

| Index | Feature |
|---|---|
| 0–28 | 1 if the mover has a piece on compact square `c` (index `c`) |
| 29–57 | 1 if the opponent has a piece on compact square `c` (index `29 + c`) |
| 58–63 | one-hot number of the mover's pieces borne off (0 to 5) |
| 64–69 | one-hot number of the opponent's pieces borne off (0 to 5) |
| 70 | mover's remaining distance `sum(31 - s) / 100` over its squares `s` |
| 71 | opponent's remaining distance, likewise |

## Integrity manifests

A manifest says what a database, a training-data file or a network consists of, how it
was made, and the size and SHA-256 of each of its files. The manifest of a directory is
`MANIFEST.json` in it; that of a file `NAME.ext` is `NAME.manifest.json` beside it, which
may also cover other files of that name (a network's `.bin` and `.pt`). `senet` (Rust,
`senet_core::manifest`) and Python (`senet.manifest`) write and check the same format:

```
senet verify db/kendall5
python -m senet.manifest verify db/kendall5 runs/train.bin models/senet_net.bin
```

A manifest is a JSON object:

| Key | Value |
|---|---|
| `manifest` | the format version, `1`; a reader refuses any other |
| `kind`, `format` | `"database"` and `"senet-db/1"`, `"training-data"` and `"senet-records/1"`, or `"network"` and `"snn1/1"` |
| `rules` | `"kendall5"` |
| `created_utc` | when it was written, as `2026-10-04T02:54:08+00:00` |
| `created_by` | `command` (the command line, its arguments quoted as a POSIX shell reads them back), `commit` (the source commit, with `-dirty` if the work tree had changes; `null` outside git), `engine` (see below) and, from Python, `environment` |
| `settings` | what the maker was asked for, or `null` if that was not recorded |
| `inputs` | the artifacts it was made from, each as `{"path", "manifest_sha256"}` (paths with `/`), or `null` |
| `contents` | facts about what it holds (records, layers, errors, ...), or `null` |
| `notes` | remarks, such as that the manifest was written after the fact |
| `files` | `[{"name", "bytes", "sha256"}]`: plain file names in the manifest's directory, the SHA-256 in lowercase hex |

`engine` is the engine's build (`senet.build_info()` in Python, `senet_build_info` in the
C ABI): `version`, `target` (as `x86_64-windows`), `optimized` and `cpu_features` (those
the build was compiled to use).

Verifying checks that every listed file is a plain file name (no directory, drive or
`:`), present, a regular file, and of its size and SHA-256, and for a database that its
directory holds no layer file (`L*.f32`) the manifest leaves out; it reports every problem
it finds. Comparing sizes only (`--sizes-only`, in both programs)
is quick and finds missing and truncated files. A writer removes an artifact's manifest before replacing its
files and writes the manifest last, so a manifest never vouches for files being changed.

`senet manifest --db DIR` and `senet manifest --data FILE` write the manifest of an
existing database or training-data file, with the note that it was written after the
fact, and `distill --evaluate NET --record` that of an existing network, with its errors
measured anew: they vouch for the files as they are, not for how they were made.
`manifest --data` refuses a file whose manifest would replace another file's, as
`gen-data` does.
