"""
Puts the project root on sys.path, so the scripts in myo/ can import the shared
modules (config, pipeline, agents...) when run as `python myo/<script>.py`.

Imported for its side effect, before any project module.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
