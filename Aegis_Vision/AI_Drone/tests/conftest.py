"""Pytest configuration — adds project root to sys.path for imports."""
import sys
from pathlib import Path

# Add project root to path so all modules can be imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
