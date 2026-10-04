# Senet — "Modern Kendall" rules, 5 pieces (ruleset id `kendall5`)

This is the exact, normative ruleset the engine, solver, bots, and UI implement.
Senet's original rules are lost; this follows the reconstruction by Timothy Kendall
(*Passing Through the Netherworld: The Meaning and Play of Senet, an Ancient Egyptian
Funerary Game*, Kirk Game Company, 1978), which most modern sets and online versions use,
with every ambiguity resolved explicitly (see "Interpretation notes").

## Board

* 30 squares ("houses") numbered **1–30**, played along an S-shaped track:

  ```
  row 1:   1  2  3  4  5  6  7  8  9 10   (left → right)
  row 2:  20 19 18 17 16 15 14 13 12 11   (right → left)
  row 3:  21 22 23 24 25 26 27 28 29 30   (left → right)
  ```

* Both players move in the same direction, from low numbers to high numbers.
* A piece that leaves the board is **borne off** (think of it as square 31) and never returns.

Special squares:

| Square | Name                    | Effect |
|-------:|-------------------------|--------|
| 15     | House of Rebirth        | Return point for pieces drowned on 27. Otherwise ordinary. |
| 26     | House of Beauty         | Every piece **must land exactly on 26**; no move may jump past it. |
| 27     | House of Water          | A piece that lands here is immediately sent back to 15 (see below). It is therefore **always empty**. |
| 28     | House of Three Truths   | Final square: the piece can only leave by bearing off with exactly **3**. |
| 29     | House of Re-Atoum       | Final square: the piece can only leave by bearing off with exactly **2**. |
| 30     | House of Horus          | Final square: the piece can only leave by bearing off with exactly **1**. |

## Pieces and setup

* Two players, **White** and **Black**, each with **5 pieces**.
* White starts on squares **1, 3, 5, 7, 9**. Black starts on squares **2, 4, 6, 8, 10**.
* **White moves first.**

## Throwing sticks

Four two-sided sticks; the throw equals the number of light sides up, except that
zero light sides counts as **5**. With fair sticks:

| Throw | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Probability | 4/16 | 6/16 | 4/16 | 1/16 | 1/16 |

## Turn structure

1. The player to move throws the sticks.
2. If they have at least one legal move for that throw, they **must** make exactly one
   (they choose which).
3. If they moved and the throw was **1, 4 or 5**, they throw again (back to step 1).
   If the throw was **2 or 3**, the turn passes to the opponent.
4. If they have **no** legal move, the turn passes to the opponent immediately
   (even if the throw was 1, 4 or 5).
5. A player wins **immediately** when they bear off their last piece.

## Definitions (from the point of view of the player moving)

* **Friendly** pieces belong to the mover, **enemy** pieces to the opponent.
* An enemy piece on square `s` is **protected** if another enemy piece occupies
  square `s-1` or `s+1`. (Square 27 is always empty, so 26 and 28 are never adjacent
  in practice.)
* A **blockade** is any run of **three or more** enemy pieces on consecutive squares.
  Every square in such a run is a *blockade square*.

## Legal moves for a throw `t`

A move always moves a single friendly piece. Let `a` be its square.

### Forward moves

1. If `a` is a final square (28, 29, 30): the only possible move is bearing off, which is
   legal iff `a + t = 31` (28 needs 3, 29 needs 2, 30 needs 1). Nothing else is possible
   for that piece.
2. Otherwise the destination is `d = a + t`.
3. **House of Beauty:** if `a < 26` and `d > 26`, the move is illegal.
4. **Blockade:** if any square strictly between `a` and `d` is a blockade square, the move
   is illegal. (This includes bearing off from 26 with a 5 over an enemy blockade on
   28-29-30.)
5. If `d = 31` (only possible from 26 with a 5): the piece is **borne off**.
6. If `d = 27` (only possible from 26 with a 1): **House of Water** — the piece is placed on
   square 15 if it is empty; otherwise on the highest-numbered empty square below 15
   (14, then 13, …). This never captures.
7. If `d` holds a friendly piece: illegal.
8. If `d` holds an enemy piece: illegal if that piece is protected; otherwise the two
   pieces **swap** (the mover's piece goes to `d`, the enemy piece goes to `a`).
9. Otherwise (`d` empty): the piece moves to `d`.

### Backward moves

Backward moves are allowed **only if no forward move exists** (for any friendly piece)
with this throw. Then, for a friendly piece on `a` with `a ≤ 26` (pieces on final squares
never move backward):

1. `d = a - t`; illegal if `d < 1`.
2. **Blockade:** illegal if any square strictly between `d` and `a` is a blockade square.
3. Then rules 7–9 above apply (friendly → illegal; protected enemy → illegal; unprotected
   enemy → swap, the enemy piece goes to `a`; empty → move).

### No move

If there is no legal forward move and no legal backward move, the turn is forfeited
(rule 4 of the turn structure).

## Interpretation notes (where sources disagree)

* **Swap capture** (most modern sets) rather than "send to first empty square behind".
* **Protection by adjacency** applies on every square, including 28–30.
* **Final squares 28/29/30 are immobile** except for bearing off with the exact throw
  (Kendall's original description); pieces there cannot step to another final square and
  cannot move backward.
* **Bearing off** is only possible from 26 (with 5), 28 (3), 29 (2), 30 (1).
* **Forfeit ends the whole turn**, even after a 1/4/5.
* There is no special "first throw must be 1" opening rule; White simply moves first.

## Position representation used throughout the code

* A position is stored **from the point of view of the player about to throw**:
  `me` = bitmask of the mover's occupied squares, `opp` = bitmask of the opponent's.
  Bit `i` set ⇔ square `i` occupied (bits 1..30; bit 0 and bit 27 are always 0).
* Pieces borne off = 5 − popcount.
* The **value** of a position is the probability that the player about to throw wins,
  assuming both sides play optimally from then on.
