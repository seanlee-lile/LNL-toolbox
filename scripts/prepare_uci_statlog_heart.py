from __future__ import annotations

"""Download and verify the UCI Statlog Heart raw benchmark."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile


URL = "https://archive.ics.uci.edu/static/public/145/statlog%2Bheart.zip"
ARCHIVE_SHA256 = "04e44018af4f4b32008029fbfb1079e112d3b372bb032dd1ae361c74d7528b2c"
RAW_SHA256 = "f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9"
DEFAULT_OUTPUT = Path("data/uci/statlog-heart/heart.dat")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare(
    output: Path,
    archive: Path | None = None,
    raw_source: Path | None = None,
) -> Path:
    if archive is not None and raw_source is not None:
        raise ValueError("provide at most one of --archive and --raw")
    if raw_source is not None:
        raw = raw_source.read_bytes()
    elif archive is None:
        with urlopen(URL, timeout=120) as response:
            archive_bytes = response.read()
    else:
        archive_bytes = archive.read_bytes()
    if raw_source is None:
        if _sha256(archive_bytes) != ARCHIVE_SHA256:
            raise ValueError("UCI Statlog Heart archive SHA-256 mismatch")
        with tempfile.TemporaryDirectory(prefix="uci-statlog-heart-") as directory:
            archive_path = Path(directory) / "heart.zip"
            archive_path.write_bytes(archive_bytes)
            with zipfile.ZipFile(archive_path) as bundle:
                raw = bundle.read("heart.dat")
    if _sha256(raw) != RAW_SHA256:
        raise ValueError("UCI Statlog Heart raw-file SHA-256 mismatch")
    if output.exists():
        if _sha256(output.read_bytes()) != RAW_SHA256:
            raise FileExistsError(
                f"refusing to overwrite a different existing file: {output}"
            )
        return output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".dat.pending")
    temporary.write_bytes(raw)
    temporary.replace(output)
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--raw", type=Path)
    arguments = parser.parse_args()
    print(prepare(arguments.output, arguments.archive, arguments.raw))


if __name__ == "__main__":
    main()
