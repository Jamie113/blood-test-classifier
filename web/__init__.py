"""FastAPI front-end package.

The analysis layer (analysis.py, gmm.py, parsing.py, thresholds.py,
unit_conversions.py, column_map.py) lives at the project root rather than
inside this package, so we make it importable for every submodule by
prepending the project root to sys.path on package load. A proper
src-layout with pyproject.toml is the longer-term fix (Tier 2 of the
hardening plan).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
