"""Distillation of the perfect-play database into a small neural network (needs PyTorch).

* ``python -m senet_train.distill``: train the network and export it (SNN1 ``.bin`` + ``.pt``).
* ``python -m senet_train.check_net``: parity of the PyTorch, Rust and browser forward passes.
"""
