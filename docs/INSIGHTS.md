# What the solution says

Ruleset: Modern Kendall, 5 pieces ([RULES.md](../RULES.md)). The raw results are in [INSIGHTS.json](INSIGHTS.json).

## First-player advantage

With perfect play, **White, who throws first, wins 50.2504%** of games, and Black wins 49.7496%.
Moving first outweighs Black's head start (front piece on 10 vs 9): White leads by 0.50 points.

## The best opening move for every throw

Win probability for White after each legal first move. ★ = optimal.

| Throw (prob.) | Move | White wins |
|---|---|---:|
| 1 (4/16) | 1→2 swap ★ | 51.191% |
|  | 3→4 swap | 50.747% |
|  | 5→6 swap | 50.052% |
|  | 9→10 swap | 49.924% |
|  | 7→8 swap | 49.620% |
| 2 (6/16) | 9→11 ★ | 48.703% |
| 3 (4/16) | 1→4 swap ★ | 51.090% |
|  | 3→6 swap | 50.685% |
|  | 5→8 swap | 49.805% |
|  | 9→12 | 48.697% |
|  | 7→10 swap | 48.500% |
| 4 (1/16) | 7→11 ★ | 50.349% |
|  | 9→13 | 50.261% |
| 5 (1/16) | 1→6 swap ★ | 52.318% |
|  | 3→8 swap | 50.697% |
|  | 7→12 | 50.418% |
|  | 9→14 | 50.203% |
|  | 5→10 swap | 49.341% |

## Perfect self-play (20,000 games)

Ranges are 95% intervals.

* White won 50.18% (49.48–50.87%); the database's value is 50.25%.
* A game lasts 198 throws on average, with 132 real decisions (two or more legal moves) and 3.5 forfeited throws.
* Moves played: 71.4% ordinary, 13.8% swaps, 8.4% forced backward, 2.0% into the House of Water, 4.4% bearing off.
* Comebacks: winners had been below a 25% chance in 20.3% of games (19.8–20.9%), below 10% in 2.3% (2.1–2.5%), below 5% in 0.10% (0.07–0.16%), never below 1%.

## Provenance

* Generated 2026-10-04T20:26:48+00:00 by `python -m senet.insights --db db/kendall5` at commit `9f8ac2aa17fb5ef6304e19aded162dc804f41042`.
* Machine: Intel(R) Core(TM) i9-14900K, 32 logical CPUs, 64 GiB; Windows-11-10.0.26200-SP0; Python 3.12.10.
* Engine: senet 0.1.0 for x86_64-windows, optimized build using no CPU features beyond the baseline.
* Database: `db/kendall5`, manifest SHA-256 `251bbefe891c6475686360eb8613623ef7b3f8f4ceba2dc40d958f849533ee2f`.
* Settings: games = 20000, seed = 2026, optimal_tolerance = 1e-09.
