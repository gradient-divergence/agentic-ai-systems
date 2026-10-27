import subprocess
import sys
from pathlib import Path

from setuptools.config.pyprojecttoml import read_configuration


def test_spine_imports_after_directory_move() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import spine.run_spine; import spine.ml.run_gbm_baseline; "
            "import spine.ml.run_embedding_clustering; import spine.rl.run_policy_eval; "
            "import spine.ops.run_capstone",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_install_declares_companion_packages() -> None:
    config = read_configuration(str(Path(__file__).resolve().parents[1] / "pyproject.toml"))
    packages = config["tool"]["setuptools"]["packages"]
    assert {"agents", "spine", "spine.ml", "demos"} <= set(packages)
    assert "tests" not in packages
