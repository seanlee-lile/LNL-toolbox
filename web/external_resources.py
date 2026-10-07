"""External files used by paper configurations, with fixed download sources."""

from functools import lru_cache
import hashlib
from pathlib import Path
import ssl
import tempfile
import threading
from urllib.request import urlopen
import zipfile


_LOCK = threading.RLock()
_JOBS = {}


@lru_cache(maxsize=64)
def _digest(path, size, modified):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid(path, checksum):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if not checksum:
        return True
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns).startswith(checksum)


def resources(config, root):
    """Inspect actual configured dependencies; generated inputs aren't downloads."""
    root = Path(root).resolve()
    items = []

    def add(key, title, path, url, checksum="", note="", archive_member=None):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        ready = _valid(path, checksum)
        with _LOCK:
            job = dict(_JOBS.get(str(path), {}))
        pending_status = job.get("status")
        if pending_status not in {"downloading", "failed"}:
            pending_status = "missing"
        problem = "" if ready else job.get("error", "")
        if not ready and path.is_file() and checksum and not problem:
            problem = "文件校验未通过，请检查下载是否完整：" + str(path)
        items.append(dict(id=key, title=title, path=str(path), url=url,
                          checksum=checksum, archive_member=archive_member,
                          ready=ready, status="ready" if ready else pending_status,
                          error=problem, note=note,
                          downloadable=bool(url)))

    feature = config.get("dld", {}).get("feature_extractor", {})
    if feature.get("source") == "external_checkpoint" and feature.get("external", {}).get("adapter") == "torchvision_resnet34_imagenet1k_v1":
        import torch
        add("dld-resnet34", "ResNet-34 预训练权重", Path(torch.hub.get_dir()) / "checkpoints/resnet34-b627a593.pth",
            "https://download.pytorch.org/models/resnet34-b627a593.pth", "b627a593",
            "用于 DLD 提取图像特征，约 83 MB。自行下载后保留原文件名。")
    data = config.get("data", {})
    if data.get("name") == "uci_statlog_heart":
        from scripts.prepare_uci_statlog_heart import URL, RAW_SHA256
        add("uci-heart", "UCI Statlog Heart 数据", data.get("path", "data/uci/statlog-heart/heart.dat"),
            URL, RAW_SHA256, "自行下载后解压，将 heart.dat 放到下面的路径。", "heart.dat")
    noise = config.get("noise", {})
    if config.get("method") == "cal" and noise.get("name") == "external_torch" and noise.get("path"):
        filename = Path(noise["path"]).name
        if filename == "IDN_0.2_C10.pt":
            add("cal-idn20", "CIFAR-10 实例相关噪声标签", noise["path"],
                "https://raw.githubusercontent.com/UCSC-REAL/CAL/main/datasets/cifar-10-batches-py/noise_label/IDN_0.2_C10.pt",
                "12634bf5405a6ae17c281a60a5cf7130640e9ec858f328be73b1d4fc03f57cee",
                note="作者发布的 20% 实例相关噪声标签；仅在配置使用该标签文件时需要。")
    provider = config.get("pipeline", {}).get("weight_provider", {})
    if provider.get("name") == "mentornet" and provider.get("artifact_path"):
        add("mentor-artifact", "MentorNet 权重", provider["artifact_path"], "",
            note="由本仓库的 MentorNet 准备和训练流程生成；使用下方准备入口。没有可直接下载的对应权重。")
        from lnl_toolbox.catalog import mentornet_preparation_status, resolve_config_paths
        preparation = mentornet_preparation_status(resolve_config_paths(config, root), root)
        items[-1]["ready"] = bool(preparation["artifact_ready"])
        items[-1]["status"] = "ready" if items[-1]["ready"] else "missing"
        items[-1]["error"] = preparation.get("artifact_error") or ""
    return items


def _context():
    # Some Windows certificate stores contain malformed entries. Use the
    # installed CA bundle when available, without disabling TLS verification.
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download(item):
    target = Path(item["path"])
    pending = None
    try:
        if target.exists():
            if _valid(target, item["checksum"]):
                return
            raise ValueError("目标文件已存在但校验未通过，请先移走该文件再下载：" + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".pending", delete=False) as stream:
            pending = Path(stream.name)
            with urlopen(item["url"], timeout=60, context=_context()) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        if item["archive_member"]:
            with zipfile.ZipFile(pending) as bundle:
                raw = bundle.read(item["archive_member"])
            pending.write_bytes(raw)
        if not _valid(pending, item["checksum"]):
            raise ValueError("下载文件校验失败，请尝试自行下载")
        if item["id"] == "cal-idn20":
            import torch
            labels = torch.load(pending, map_location="cpu", weights_only=True)
            if not all(key in labels and len(labels[key]) == 50000 for key in ("clean_label_train", "noise_label_train")):
                raise ValueError("CAL 标签文件缺少正确的 clean/noisy 标签")
        # Hard-link creation is atomic and refuses to replace an existing file.
        import os
        os.link(pending, target)
        with _LOCK:
            _JOBS[str(target)] = {"status": "ready"}
    except Exception as exc:
        with _LOCK:
            _JOBS[str(target)] = {"status": "failed", "error": str(exc)}
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def start_download(item):
    if not item["downloadable"]:
        raise ValueError("此资源需要本地生成，请使用论文栏中的准备入口")
    with _LOCK:
        if item["ready"] or _JOBS.get(item["path"], {}).get("status") == "downloading":
            return
        _JOBS[item["path"]] = {"status": "downloading"}
    threading.Thread(target=_download, args=(item,), daemon=True).start()
