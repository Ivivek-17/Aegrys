"""Export the dashboard as a static site for hosting on a domain.

The same files serve both purposes. Deployed statically there is no control API,
so the UI probes `/api/status`, gets nothing, and switches itself to DOCS mode:
setup, architecture and benchmark screens work fully; the control screens explain
that process control requires a local install.

    python scripts/build_site.py          -> ./dist
    python scripts/build_site.py --serve  -> preview it as a visitor would

Deploy `dist/` to GitHub Pages, Netlify, Cloudflare Pages, or any static host.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "aegrys" / "web" / "static"
DIST = ROOT / "dist"


def build() -> Path:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(STATIC, DIST)

    # GitHub Pages otherwise runs the output through Jekyll, which drops
    # underscore-prefixed paths and can mangle assets.
    (DIST / ".nojekyll").write_text("")

    # SPA-style fallback so deep links like /#setup survive a hard refresh on
    # hosts that support it.
    shutil.copy(DIST / "index.html", DIST / "404.html")

    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"built {DIST}")
    for f in sorted(DIST.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(DIST)}  ({f.stat().st_size/1024:.1f} KB)")
    print(f"\ntotal {total/1024:.1f} KB — no build step, no dependencies")
    return DIST


def serve(port: int) -> None:
    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(DIST))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"\npreviewing static build at http://127.0.0.1:{port}")
        print("(this is DOCS mode — exactly what a visitor to your domain sees)")
        print("Ctrl-C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true", help="preview after building")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()
    build()
    if a.serve:
        serve(a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
