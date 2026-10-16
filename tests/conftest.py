"""Make the repo root importable (src/, serverv2/) and keep tests offline and trace-free."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):   # tracing becomes a no-op
    os.environ.pop(k, None)
