"""Test environment.

Set before anything can import torch: on Apple Silicon, MPS does not implement
every operator, and without this flag an unimplemented op aborts the process
instead of falling back to CPU. It has to be in place before the first MPS
tensor is created, so this module does it at import time rather than in a
fixture.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
