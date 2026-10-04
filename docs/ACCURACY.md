# How accurate the database is

Every position has its own stored value: nothing is sampled, and no model stands in for
the database. The values are computed by iteration in floating point, though, so each one
differs a little from the true probability.

**No value is believed to be more than about 1e-6 from the true probability.** That is an
estimate from the measurements below. It is not a proven bound: no global error bound has
been derived.

## Where the error comes from

* **Rounding.** The database stores `float32` values, whose steps on [0, 1) are 6e-8 or
  less. The 5v5 layer was solved in 24-bit fixed point, in steps of 6e-8.
* **Stopping.** Value iteration approaches the solution without reaching it. The solver
  stops a group of layers once a whole sweep changes no value by more than a tolerance.
  The six groups with 7 to 10 pieces used 3e-7 and stopped at last changes of 2.4e-7 to
  3.0e-7. The nine smaller groups used 1e-7 and stopped at 6e-8 to 9e-8. The last change
  is not the remaining error. At the end of each large group the largest change was
  shrinking by a factor of 0.68 to 0.78 per sweep; had that continued, the values would
  have moved by a further 5e-7 to 1.1e-6.
* **Inherited error.** A layer's values depend on the layers with one piece fewer, so
  their errors carry over.

`db/kendall5/meta.json` records each group's sweeps and last change. The tolerance of the
large groups was the solver's default and is not in their records. The records of the
small groups were added afterwards from the solve's log, and say so.

## What was measured

* **An independent reference.** A pure-Python solver in `float64`, run until no value
  changes by more than 1e-13, agrees with the database within 1.5e-7 on all 3,917,900
  positions with up to 5 pieces in total (1.8e-8 to 3.5e-8 on average, by layer). There
  is no independent reference for the larger layers.

  ```bash
  python -m senet_ref.check_db db/kendall5 --solve
  ```

* **Bellman residuals.** `senet check` recomputes |T(V) − V| on 200,000 random positions
  of every layer, and on every position of the layers smaller than that. The worst
  residual over all 25 layers is 1.3e-7. A small residual shows the table is nearly
  self-consistent. In a game with cycles it does not bound the error, which can be larger
  by a factor that grows as the iteration converges more slowly.

  ```bash
  target/release/senet check --db db/kendall5
  ```

* **The opening.** The value of the opening position stayed between 0.5025043 and
  0.5025044 over the last 12 sweeps of the 5v5 solve.
* **Play.** White's share of the wins in games of the perfect player against itself
  agrees with the database's 50.25%, within the sampling error of the games played
  ([INSIGHTS.md](INSIGHTS.md) has the numbers).

## What it means for play

If every value is within ε of the truth, the move the perfect player chooses is worth at
most 2ε less than the best move. It can only miss the best move when two moves' values
differ by less than about 2e-6. The benchmarks therefore count a decision as an error
only when it gives away more than 2e-6.

## What else is checked

* **The rules.** Three implementations (Rust, a Python reference written only from
  [RULES.md](../RULES.md), and the JavaScript engine of the web page) agree move for move
  on 1,000,000 random positions.
* **The network's arithmetic.** The Rust and JavaScript forward passes agree with PyTorch
  on 20,000 random positions ([NETWORK.md](NETWORK.md)).
* **The files.** `senet verify db/kendall5` checks every file of a database against the
  size and SHA-256 in its manifest ([FORMATS.md](FORMATS.md)).

`python tools/check_all.py --full` runs all of these.
