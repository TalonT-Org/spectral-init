import subprocess
import sys
from pathlib import Path

# Make the tests/ directory importable so test modules can import fixture_utils.
sys.path.insert(0, str(Path(__file__).parent))

_GENERATE_SCRIPT = Path(__file__).parent / "generate_fixtures.py"


def run_fixture_pipeline(
    datasets: list[str], outdir: str, extra_args: list[str] | None = None
) -> None:
    """Run generate_fixtures.py for the given datasets into outdir."""
    cmd = [sys.executable, str(_GENERATE_SCRIPT), "--output-dir", outdir, "--datasets"] + datasets
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"CMD: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
