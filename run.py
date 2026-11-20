"""Start Reso-Noise:  python run.py [--host 127.0.0.1] [--port 5050]"""
from __future__ import annotations

import argparse
import logging

from resonoise.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Reso-Noise audio analysis studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app()
    print(f"\n  Reso-Noise is running at  http://{args.host}:{args.port}\n  (microphone recording works on localhost or HTTPS)\n")
    try:
        from waitress import serve
        if not args.debug:
            serve(app, host=args.host, port=args.port, threads=4, max_request_body_size=60 * 1024 * 1024)
            return
    except ImportError:
        pass
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
