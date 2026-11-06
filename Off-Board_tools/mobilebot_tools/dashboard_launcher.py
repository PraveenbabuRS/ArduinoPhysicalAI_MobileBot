"""Thin launcher so `mobilebot-dashboard` works like any other console script,
instead of requiring `streamlit run <path found by hand>`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        raise SystemExit(
            "streamlit is not installed. Run: pip install -e '.[dashboard]'")

    dashboard_path = str(Path(__file__).parent / 'dashboard.py')
    sys.argv = ['streamlit', 'run', dashboard_path, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == '__main__':
    main()
