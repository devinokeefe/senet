"""Build dist/senet-portable.html: one self-contained file (UI + JS rules engine +
distilled network weights) that plays Senet with no server and no database.

    python tools/build_portable.py [--net models/senet_net.bin] [--out dist/senet-portable.html]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))  # if the project is not installed

from senet_train import snn1

ROOT = Path(__file__).resolve().parents[1]


def net_to_js(path: Path) -> str:
    """A script defining window.SENET_NET, the network with base64-encoded float32 weights;
    ValueError if the file is not a network that the SNN1 format allows."""
    js_layers = [
        {
            "in": w.shape[1],
            "out": w.shape[0],
            "w": base64.b64encode(w.tobytes()).decode(),
            "b": base64.b64encode(b.tobytes()).decode(),
        }
        for w, b in snn1.read(path.read_bytes())
    ]
    return "window.SENET_NET = " + json.dumps({"format": "senet-mlp-v1", "layers": js_layers}) + ";"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", type=Path, default=ROOT / "models" / "senet_net.bin")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "senet-portable.html")
    args = ap.parse_args(argv)
    web = ROOT / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    css = (web / "style.css").read_text(encoding="utf-8")
    try:
        net_js = net_to_js(args.net)
    except OSError as e:
        raise SystemExit(f"cannot read the network: {e}") from None
    except ValueError as e:
        raise SystemExit(f"{args.net}: {e}") from None
    scripts = [
        net_js,
        (web / "portable" / "senet-engine.js").read_text(encoding="utf-8"),
        (web / "portable" / "local-api.js").read_text(encoding="utf-8"),
        (web / "app.js").read_text(encoding="utf-8"),
    ]

    def swap(old: str, new: str) -> None:
        nonlocal html
        if old not in html:
            raise SystemExit(f"web/index.html no longer contains {old!r}")
        html = html.replace(old, new)

    swap('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    swap('<script src="app.js"></script>', "\n".join(f"<script>\n{s}\n</script>" for s in scripts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
