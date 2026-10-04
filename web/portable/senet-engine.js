"use strict";
// Standalone Senet engine for the browser (no server): exact rules (RULES.md),
// the distilled neural network, and expectimax search over stick throws.
// Positions are {me, opp} bit masks (bit i = square i) from the mover's point of view.
// Verified move-for-move against the Rust engine (see tools/check_js_rules.mjs).

(function (root) {
  const OFF = 31, BEAUTY = 26, WATER = 27, REBIRTH = 15;
  const THROW_PROBS = [0, 4 / 16, 6 / 16, 4 / 16, 1 / 16, 1 / 16];
  const EXTRA = [false, true, false, false, true, true];

  // Masks use bits 1..30 only, so they are non-negative int32 values.
  const bit = (s) => 1 << s;
  const has = (m, s) => ((m >>> s) & 1) === 1;

  function squares(m) { const r = []; for (let s = 1; s <= 30; s++) if (has(m, s)) r.push(s); return r; }
  function maskOf(list) { return list.reduce((m, s) => m + bit(s), 0); }

  function protectedSet(opp) {
    const p = new Set();
    for (let s = 1; s <= 30; s++) if (has(opp, s) && ((s > 1 && has(opp, s - 1)) || (s < 30 && has(opp, s + 1)))) p.add(s);
    return p;
  }
  function blockadeSet(opp) {
    const b = new Set();
    for (let s = 1; s + 2 <= 30; s++) if (has(opp, s) && has(opp, s + 1) && has(opp, s + 2)) { b.add(s); b.add(s + 1); b.add(s + 2); }
    return b;
  }
  function pathBlocked(block, lo, hi) {
    for (let s = lo + 1; s < hi; s++) if (block.has(s)) return true;
    return false;
  }

  function land(me, opp, prot, a, d, back) {
    if (has(me, d)) return null;
    if (has(opp, d)) {
      if (prot.has(d)) return null;
      return { from: a, to: d, kind: "swap", back, me: me - bit(a) + bit(d), opp: opp - bit(d) + bit(a) };
    }
    return { from: a, to: d, kind: "move", back, me: me - bit(a) + bit(d), opp };
  }
  // Landing on 27 (only from 26, forward): back to 15, or the highest empty square below it.
  function water(me, opp, a) {
    const rest = me - bit(a);
    let s = REBIRTH;
    while (has(rest, s) || has(opp, s)) s--;
    return { from: a, to: s, kind: "water", back: false, me: rest + bit(s), opp };
  }

  /** All legal moves for throw t, sorted by origin square. */
  function legalMoves(me, opp, t) {
    const prot = protectedSet(opp), block = blockadeSet(opp);
    const out = [];
    const mine = squares(me);
    for (const a of mine) {
      if (a >= 28) {
        if (a + t === OFF) out.push({ from: a, to: OFF, kind: "off", back: false, me: me - bit(a), opp });
        continue;
      }
      const d = a + t;
      if (a < BEAUTY && d > BEAUTY) continue;
      // Here a <= 26 (27 is always empty), so d <= 31: only 26 + 5 reaches square 31.
      if (pathBlocked(block, a, d)) continue;
      if (d === OFF) out.push({ from: a, to: OFF, kind: "off", back: false, me: me - bit(a), opp });
      else if (d === WATER) out.push(water(me, opp, a));
      else { const m = land(me, opp, prot, a, d, false); if (m) out.push(m); }
    }
    if (out.length) return out;
    for (const a of mine) {
      if (a > BEAUTY || a <= t) continue;
      const d = a - t;
      if (pathBlocked(block, d, a)) continue;
      const m = land(me, opp, prot, a, d, true);
      if (m) out.push(m);
    }
    return out;
  }

  // ---- Neural network ----
  // JSON form "senet-mlp-v1" of an SNN1 network (docs/FORMATS.md), made by net_to_js in
  // tools/build_portable.py: layers of {in, out, w, b}, weights and biases as base64
  // little-endian float32. It is held to the format's limits, as in senet_core::net.
  const NET_FORMAT = "senet-mlp-v1";
  const N_INPUTS = 72, MAX_LAYERS = 16, MAX_WIDTH = 4096;
  // The largest feature value: a remaining distance, at most (30 + 29 + 28 + 27 + 26) / 100.
  const MAX_FEATURE = 1.4, F32_MAX = 3.4028234663852886e38;
  function floats(base64, what) {
    const bin = atob(base64), bytes = new Uint8Array(bin.length);
    if (bin.length % 4) throw new Error(`${what}: ${bin.length} bytes is not a whole number of float32 values`);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function loadNet(json) {
    if (json.format !== NET_FORMAT) throw new Error(`network format ${json.format}, expected ${NET_FORMAT}`);
    const n = json.layers.length;
    if (!(n >= 1 && n <= MAX_LAYERS)) throw new Error(`${n} layers; a network has 1..${MAX_LAYERS}`);
    const net = [];
    let nInExpected = N_INPUTS;
    for (const [k, l] of json.layers.entries()) {
      const nIn = l.in, nOut = l.out;
      if (nIn !== nInExpected || !Number.isInteger(nOut) || nOut < 1 || nOut > MAX_WIDTH) {
        throw new Error(`layer ${k} is ${nIn} -> ${nOut}; expected ${nInExpected} inputs and 1..${MAX_WIDTH} outputs`);
      }
      const w = floats(l.w, `layer ${k} weights`), b = floats(l.b, `layer ${k} biases`);
      if (w.length !== nIn * nOut || b.length !== nOut) {
        throw new Error(`layer ${k} has ${w.length} weights and ${b.length} biases, not ${nIn * nOut} and ${nOut}`);
      }
      if (!w.every(Number.isFinite) || !b.every(Number.isFinite)) throw new Error(`layer ${k} has a parameter that is not finite`);
      net.push({ nIn, nOut, w, b });
      nInExpected = nOut;
    }
    if (nInExpected !== 1) throw new Error(`the network has ${nInExpected} outputs; it must have one`);
    // Bounds on the absolute values of each layer's inputs: the features, then the previous
    // layer's outputs. Sums of terms within half the range of float32 cannot overflow it.
    let bound = new Float64Array(N_INPUTS).fill(MAX_FEATURE);
    for (const [k, L] of net.entries()) {
      const next = new Float64Array(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let sum = 0;
        for (let i = 0; i < L.nIn; i++) sum += Math.abs(L.w[o * L.nIn + i]) * bound[i];
        next[o] = Math.abs(L.b[o]) + sum;
      }
      if (next.some((x) => x > F32_MAX / 2)) throw new Error(`layer ${k}'s outputs could overflow float32`);
      bound = next;
    }
    return net;
  }
  // Feature index of a square: 1..26 -> 0..25, 28..30 -> 26..28 (27 is always empty).
  const compactIndex = (s) => (s <= 26 ? s - 1 : s - 2);
  function features(me, opp) {
    const f = [];
    let md = 0, od = 0, mc = 0, oc = 0;
    for (let s = 1; s <= 30; s++) {
      if (has(me, s)) { f.push([compactIndex(s), 1]); md += 31 - s; mc++; }
      if (has(opp, s)) { f.push([29 + compactIndex(s), 1]); od += 31 - s; oc++; }
    }
    f.push([58 + Math.max(0, 5 - mc), 1], [64 + Math.max(0, 5 - oc), 1], [70, md / 100], [71, od / 100]);
    return f;
  }
  /** The network's output before the sigmoid, for a game in progress. */
  function netLogit(net, me, opp) {
    const l0 = net[0];
    let h = Float32Array.from(l0.b);
    for (const [i, x] of features(me, opp)) for (let o = 0; o < l0.nOut; o++) h[o] += x * l0.w[o * l0.nIn + i];
    for (let k = 1; k < net.length; k++) {
      const L = net[k], out = Float32Array.from(L.b);
      for (let o = 0; o < L.nOut; o++) {
        let acc = out[o];
        const row = o * L.nIn;
        for (let i = 0; i < L.nIn; i++) acc += L.w[row + i] * Math.max(0, h[i]);
        out[o] = acc;
      }
      h = out;
    }
    return h[0];
  }
  function netValue(net, me, opp) {
    if (me === 0) return 1;
    if (opp === 0) return 0;
    return 1 / (1 + Math.exp(-netLogit(net, me, opp)));
  }

  // ---- Search ----
  function expectimax(evalFn, me, opp, depth) {
    if (me === 0) return 1;
    if (opp === 0) return 0;
    if (depth === 0) return evalFn(me, opp);
    let v = 0;
    for (let t = 1; t <= 5; t++) {
      const moves = legalMoves(me, opp, t);
      let q;
      if (!moves.length) q = 1 - expectimax(evalFn, opp, me, depth - 1);
      else { q = 0; for (const m of moves) q = Math.max(q, childValue(evalFn, m, t, depth - 1)); }
      v += THROW_PROBS[t] * q;
    }
    return v;
  }
  function childValue(evalFn, m, t, depth) {
    if (m.me === 0) return 1;
    return EXTRA[t] ? expectimax(evalFn, m.me, m.opp, depth) : 1 - expectimax(evalFn, m.opp, m.me, depth);
  }

  root.SenetEngine = { legalMoves, loadNet, netLogit, netValue, childValue, squares, maskOf, THROW_PROBS, EXTRA };
  if (typeof module !== "undefined") module.exports = root.SenetEngine;
})(typeof window !== "undefined" ? window : globalThis);
