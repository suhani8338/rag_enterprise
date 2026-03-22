import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "src.pipeline"],
    capture_output=False,
    text=True,
)
sys.exit(result.returncode)
