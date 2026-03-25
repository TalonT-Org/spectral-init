import sys
from pathlib import Path

# Add this directory to sys.path so test modules can import generate_umap_comparisons.
# Placed in conftest.py rather than test module level to avoid module-import side effects.
sys.path.insert(0, str(Path(__file__).parent))
