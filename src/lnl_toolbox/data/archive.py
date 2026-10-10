from __future__ import annotations

"""Bounded, path-safe extraction shared by all dataset formats."""

from contextlib import contextmanager
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import zipfile


def is_dataset_archive(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))


@contextmanager
def archive_entries(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            entries = [(m.filename, m.file_size, m.is_dir(), m) for m in archive.infolist()]
            if any(stat.S_ISLNK(m.external_attr >> 16) for m in archive.infolist()):
                raise ValueError("dataset archives must not contain symbolic links")
            yield entries, archive.open
    else:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            if any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError("dataset archives must contain only files and directories")
            yield [(m.name, m.size, m.isdir(), m) for m in members], archive.extractfile


def safe_members(entries) -> list:
    if len(entries) > 100000 or sum(e[1] for e in entries) > 10 * 1024**3:
        raise ValueError("dataset archive exceeds 100000 entries or 10 GiB unpacked")
    result = []
    names = set()
    for name, size, directory, member in entries:
        relative = PurePosixPath(name.replace("\\", "/"))
        if relative.is_absolute() or any(p == ".." or ":" in p or p.endswith((".", " ")) or Path(p).is_reserved() for p in relative.parts) or relative.as_posix() == ".complete":
            raise ValueError(f"unsafe archive path: {name}")
        if not relative.parts:
            continue
        key = relative.as_posix().casefold()
        if key in names:
            raise ValueError(f"duplicate archive path: {name}")
        names.add(key)
        result.append((relative.as_posix(), size, directory, member))
    return result


def archive_layout(path: Path) -> tuple[str, ...]:
    with archive_entries(path) as (entries, _):
        return tuple(name for name, _, directory, _ in safe_members(entries) if not directory)


def extract_dataset_archive(path: Path, cache_root: Path) -> Path:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / digest.hexdigest()
    if (destination / ".complete").is_file():
        return destination
    # Extract into a private staging directory; never overwrite user datasets.
    with tempfile.TemporaryDirectory(prefix="dataset-", dir=cache_root) as temporary:
        staging = Path(temporary)
        with archive_entries(path) as (entries, opener):
            for name, _, directory, member in safe_members(entries):
                target = staging / name
                if directory:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with opener(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
        (staging / ".complete").touch()
        if not destination.exists():
            staging.rename(destination)
    return destination
