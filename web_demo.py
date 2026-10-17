#!/usr/bin/env python3
"""
SpaceTime-DSCA web demo: find a person in any video from a text description.

  python web_demo.py                       # http://127.0.0.1:8000
  python web_demo.py --port 8080 --no-browser --samples_dir /path/to/videos

The app lives in webapp/ (engine, jobs, API) and webapp/static/ (UI).
"""

import argparse
import os
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SpaceTime-DSCA video person search web demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--samples_dir", type=str, default=str(ROOT / "samples"),
                        help="Folder of sample videos shown in the UI (default: ./samples)")
    parser.add_argument("--max_upload_mb", type=int, default=1024, help="Upload size limit in MB")
    args = parser.parse_args()

    import uvicorn
    from webapp.engine import SearchEngine
    from webapp.server import create_app

    engine = SearchEngine(ROOT)
    app = create_app(engine, ROOT / "uploads", Path(args.samples_dir), args.max_upload_mb)

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 64)
    print(f"  SpaceTime-DSCA Person Search  ->  {url}")
    print(f"  Device     : {str(engine.device).upper()}")
    print(f"  Detector   : {engine.detector_backend}")
    print(f"  Checkpoint : {engine.checkpoint_info.get('name')}")
    print(f"  Samples    : {args.samples_dir}")
    print("=" * 64 + "\n")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open_new_tab(url)), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
