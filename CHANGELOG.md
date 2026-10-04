# Changelog

## 0.1.0 (2026-10-04)

The first release.

* **Solver and database.** Value iteration over all 8,560,690,670 positions of Senet under
  the Modern Kendall rules with 5 pieces per side. White, who throws first, wins 50.2504%
  under perfect play. The values are estimated to be within about 1e-6 of the true
  probabilities ([docs/ACCURACY.md](docs/ACCURACY.md)).
* **Engine.** The `senet` command, a C ABI (`senet_ffi`) and the Python package `senet`.
* **Web app.** Play the perfect player or the network in the browser, served by
  `senet serve` or as one HTML file that runs offline.
* **Distilled network.** A 72-512-256-128-1 MLP (201,729 parameters, 0.8 MB) that plays
  close to perfectly without the database ([docs/NETWORK.md](docs/NETWORK.md)).
* **Reports.** [Benchmarks](docs/BENCHMARKS.md), [strategy](docs/INSIGHTS.md) and
  [latency](docs/LATENCY.md), each with its raw results and provenance.
* **Checks.** `python tools/check_all.py`, and `--strict` as the release gate
  ([CONTRIBUTING.md](CONTRIBUTING.md)).

The database (34 GB) and the training data (2.5 GB) are not in the repository. The README
says how to make them.
