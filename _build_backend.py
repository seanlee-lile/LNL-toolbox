"""Dependency-free PEP 517 backend for local editable installs.

The project ships a traditional ``setup.py`` for metadata compatibility, but
the editable wheel is built with the standard library so pip does not need to
download setuptools from a configured package mirror.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zipfile


NAME = "lnl-toolbox"
VERSION = "0.1.0"
DIST_INFO = "lnl_toolbox-0.1.0.dist-info"


def _metadata() -> str:
    return """Metadata-Version: 2.1
Name: lnl-toolbox
Version: 0.1.0
Summary: A reproducible research toolbox for learning with noisy labels
Requires-Python: >=3.10
Requires-Dist: numpy>=1.24
Requires-Dist: pyyaml>=6.0
Provides-Extra: train
Requires-Dist: torch>=2.2; extra == \"train\"
Requires-Dist: torchvision>=0.17; extra == \"train\"
Requires-Dist: randaugment==1.0.2; extra == \"train\"
Requires-Dist: pyyaml>=6.0; extra == \"train\"
Requires-Dist: scikit-learn>=1.7,<2; extra == \"train\"
Provides-Extra: dev
Requires-Dist: build>=1.2; extra == \"dev\"
Requires-Dist: coverage[toml]>=7.6; extra == \"dev\"
Requires-Dist: pytest>=8.0; extra == \"dev\"
Requires-Dist: pytest-cov>=5.0; extra == \"dev\"
Requires-Dist: ruff>=0.9; extra == \"dev\"
Requires-Dist: tomli; python_version < \"3.11\" and extra == \"dev\"
"""


def _entry_points() -> str:
    return """[console_scripts]
lnl = lnl_toolbox.cli.main:main
lnl-clean-train = lnl_toolbox.cli.clean_train:main
lnl-inspect-data = lnl_toolbox.cli.inspect_data:main
lnl-make-noise = lnl_toolbox.cli.make_noise:main
lnl-train = lnl_toolbox.cli.train:main
"""


def _dist_info_files() -> dict[str, bytes]:
    return {
        f"{DIST_INFO}/METADATA": _metadata().encode("utf-8"),
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nGenerator: lnl-toolbox-local-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{DIST_INFO}/entry_points.txt": _entry_points().encode("utf-8"),
    }


def _wheel_name() -> str:
    return "lnl_toolbox-0.1.0-py3-none-any.whl"


def _write_wheel(wheel_directory: str, files: dict[str, bytes]) -> str:
    destination = Path(wheel_directory) / _wheel_name()
    records: list[str] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            records.append(f"{name},sha256={digest},{len(content)}")
        records.append(f"{DIST_INFO}/RECORD,,")
        archive.writestr(f"{DIST_INFO}/RECORD", ("\n".join(records) + "\n").encode("utf-8"))
    return destination.name


def _editable_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    files = _dist_info_files()
    files["lnl_toolbox_editable.pth"] = (str(root / "src") + "\n").encode("utf-8")
    return files


def get_requires_for_build_editable(config_settings=None):
    return []


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    target = Path(metadata_directory) / DIST_INFO
    target.mkdir(parents=True, exist_ok=True)
    for name, content in _dist_info_files().items():
        (target / Path(name).name).write_bytes(content)
    return DIST_INFO


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _write_wheel(wheel_directory, _editable_files())


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _write_wheel(wheel_directory, _editable_files())
