"""What the tests need beyond the package: the solved database and the trained network, which
the tests that use them are skipped without (`from conftest import needs_db, needs_net`)."""

from __future__ import annotations

import pytest

from senet._lib import ROOT

DB = ROOT / "db" / "kendall5"
NET = ROOT / "models" / "senet_net.bin"
DB_COMPLETE = all((DB / f"L{w}{b}.f32").exists() for w in range(1, 6) for b in range(1, 6))

# Skip reasons start "needs <the thing>" (see tools/check_all.py).
needs_db = pytest.mark.skipif(not DB_COMPLETE, reason="needs the solved database (db/kendall5, complete)")
needs_net = pytest.mark.skipif(not NET.exists(), reason="needs the trained network (models/senet_net.bin)")
