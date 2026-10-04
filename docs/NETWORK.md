# The network

`models/senet_net.bin` is a small neural network trained to reproduce the database's
values, so that a strong player runs without the 34 GB database: in the engine (`--net`),
from Python, and in the web page.

## What it is

A multilayer perceptron, 72-512-256-128-1 with ReLU: 201,729 parameters, 0.8 MB of
`float32`. Its input is the position as the player about to throw sees it (72 features,
listed in [FORMATS.md](FORMATS.md)). Its output, through a sigmoid, is that player's
probability of winning.

Two bots play from it. `net` plays the move that leads to the position the network values
most. `net:1` first searches one throw ahead with the network's values.

## The training data

209,476,737 positions, each with its value from the database:

* 189,476,737 from 1,000,000 games of the perfect player against itself, in which each
  side plays a random move now and then (20% of the time on average, varying from game to
  game) so that the games leave the paths of perfect play;
* 20,000,000 random positions, each drawn by choosing a layer and then a position in it.
  The small layers are therefore heavily oversampled: every 1v1 position occurs about
  985 times.

The file (2.5 GB) is not in the repository. This command makes it byte for byte, which was
checked by SHA-256 against the file the network was trained on; its manifest is
[models/training/train.manifest.json](../models/training/train.manifest.json):

```bash
target/release/senet gen-data --db db/kendall5 --games 1000000 --eps 0.2 --uniform 20000000 --seed 1 --out runs/train.bin
```

Positions recur across records: the opening is in every game. The data holds 72,286,386
distinct positions, less than 1% of the game's 8.56 billion.

## Training

2% of the records (4,189,534) were held out for validation, and the network was trained
on the other 205,287,203, which hold 71,171,383 distinct positions. Training ran for 12
epochs on an RTX 4090. An epoch took 49 to 65 s when the machine was free, about 10
minutes in all; with other jobs running, the 12 took 31 minutes
([models/training/distill.log](../models/training/distill.log)).

The network was trained by an earlier version of the training script, which did not
record its settings. Its log shows the architecture and the epochs; the rest were that
version's defaults (batch 16,384, learning rate 2e-3, weight decay 1e-5, seed 0). The
network's manifest, [models/senet_net.manifest.json](../models/senet_net.manifest.json),
was written afterwards and says so. Its errors were measured anew for it, and match the
log's.

## How close it is to the database

Mean absolute error in the win probability, in percentage points:

| Held-out records | Records | Mean error | Largest error |
|---|---:|---:|---:|
| All | 4,189,534 | 0.055 | 2.1 |
| At positions that occur in no training record | 1,118,157 | 0.081 | 2.1 |

Most held-out records are at positions that some training record shares, so the second
row is the one that shows how the network does on positions it has not seen. Those are
the data's rare positions, not a sample of the positions of play.

## How well it plays

Value error matters less than the moves it leads to.
[BENCHMARKS.md](BENCHMARKS.md) measures those over whole games against the perfect
player. A second check uses 1,631 decisions from games of an exploring perfect player,
at positions that are not in the training data, chosen by kind: a move into the House of
Water, backward moves only, an opponent's blockade, bearing off, two best moves within
0.1 percentage points of each other, and a plain sample. On every kind the network gives
away 0.008% to 0.011% of win probability per decision on average, and at most 0.94% in
one decision. With a 1-throw search it gives away 0.003% to 0.006%, and at most 0.34%.

```bash
python -m senet_train.strength
```

The decisions and the database's values for them are in `models/strength_positions.json`,
so the check needs no database. It fails if a network does markedly worse than the
committed one.

## The same arithmetic in three languages

The network runs in PyTorch (training), Rust (the engine) and JavaScript (the web page).
On 20,000 random positions the Rust and JavaScript outputs, before the sigmoid, each
differ from PyTorch's by at most 9.5e-7. The check allows 5e-6:

```bash
python -m senet_train.check_net --net models/senet_net.bin
```

## Training another

```bash
target/release/senet gen-data --db db/kendall5 --out runs/train.bin
python -m senet_train.distill --data runs/train.bin --out runs/senet_net
python -m senet_train.distill --data runs/train.bin --evaluate runs/senet_net.bin
python -m senet_train.strength --net runs/senet_net.bin
```

`gen-data` by default takes 200,000 games and 5,000,000 random positions, about 43
million records. `distill` needs PyTorch and NumPy 2 (`pip install -e ".[train]"`). It
writes the network, its PyTorch state dict and then their manifest, which records the
data, settings, split, errors and epoch times. Training is seeded, but arithmetic on a
GPU is not reproducible bit for bit: a rerun gives a network with about the same errors,
not the same parameters.
