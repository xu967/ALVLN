#!/usr/bin/env python3
"""从 HF 国内镜像续传 OpenFly 场景资产，校验通过后才移除 .part 后缀。"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

import requests

REPO = "IPEC-COMMUNITY/OpenFly_DataGen"
ENDPOINT = "https://hf-mirror.com"
API = f"{ENDPOINT}/api/datasets/{REPO}/tree/main?recursive=true&expand=true"
BASE = f"{ENDPOINT}/datasets/{REPO}/resolve/main/"
DEFAULT_OUTPUT = Path("/root/rivermind-data/uav_rgb/downloads/OpenFly_DataGen")
MAX_ATTEMPTS = 5
POLL_SECONDS = 600


class DownloadError(Exception):
    pass


class AccessDenied(DownloadError):
    pass


class RetryableError(DownloadError):
    pass


class NetworkExhausted(DownloadError):
    pass


def log(message):
    print(time.strftime("[%Y-%m-%dT%H:%M:%S%z] ") + message, flush=True)


def session():
    client = requests.Session()
    client.trust_env = False  # 用户要求国内直连，不继承代理或 .netrc 凭证。
    client.headers["Accept-Encoding"] = "identity"
    return client


def check_status(response):
    if response.status_code in (401, 403):
        raise AccessDenied(f"HTTP {response.status_code}")
    if response.status_code == 429 or response.status_code >= 500:
        raise RetryableError(f"HTTP {response.status_code}")
    if response.status_code >= 400:
        raise DownloadError(f"HTTP {response.status_code}")


def parse_manifest(rows):
    jobs = []
    seen = set()
    for row in rows:
        if row.get("type") != "file":
            continue
        name = row.get("path", "")
        if name in (".gitattributes", "README.md"):
            continue
        path = PurePosixPath(name)
        if (len(path.parts) != 2 or path.parts[0] not in ("airsim", "pcd_map", "ue")
                or ".." in path.parts or "\\" in name or name != str(path)):
            raise DownloadError("manifest contains an unexpected asset path")
        suffix = ".pcd" if path.parts[0] == "pcd_map" else ".zip"
        size = row.get("size")
        lfs = row.get("lfs") or {}
        checksum = lfs.get("oid") or row.get("oid", "")
        kind = "sha256" if lfs else "git"
        pattern = r"[0-9a-f]{64}" if lfs else r"[0-9a-f]{40}"
        if (not name.endswith(suffix) or name in seen or type(size) is not int
                or size <= 0 or not re.fullmatch(pattern, checksum)):
            raise DownloadError("manifest contains invalid size, hash or duplicate path")
        jobs.append((name, size, kind, checksum))
        seen.add(name)
    if not jobs:
        raise DownloadError("manifest contains no assets")
    return jobs


def manifest(token):
    with session() as client:
        for attempt in range(MAX_ATTEMPTS):
            try:
                with client.get(API, headers={"Authorization": f"Bearer {token}"},
                                timeout=(15, 60)) as response:
                    check_status(response)
                    if response.links.get("next"):
                        raise DownloadError("manifest is paginated; refusing an incomplete download")
                    return parse_manifest(response.json())
            except (requests.RequestException, RetryableError) as error:
                if attempt + 1 == MAX_ATTEMPTS:
                    raise NetworkExhausted(f"manifest failed: {type(error).__name__}") from None
                time.sleep(min(5 * (attempt + 1), 30))


def digest(path, kind):
    value = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git":
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def content_response(client, name, token, offset):
    url = BASE + name
    headers = {"Authorization": f"Bearer {token}"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = client.get(url, headers=headers, allow_redirects=False,
                          stream=True, timeout=(15, 60))
    if response.status_code in (301, 302, 303, 307, 308):
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise DownloadError("resolve response has no Location")
        content = urljoin(url, location)
        source, target = urlsplit(url), urlsplit(content)
        if target.scheme != source.scheme or target.username or target.password:
            raise DownloadError("unsafe content redirect")
        if (source.scheme, source.netloc) != (target.scheme, target.netloc):
            headers.pop("Authorization")  # CDN 的签名 URL 已含访问许可，不转发 HF token。
        response = client.get(content, headers=headers, stream=True, timeout=(15, 60))
    return response


def fetch(job, root, token, stop):
    name, expected, kind, checksum = job
    target = root / name
    partial = root / (name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == expected and digest(target, kind) == checksum:
            return name
        raise DownloadError(f"{name}: existing file failed verification; preserved")

    with session() as client:
        for attempt in range(MAX_ATTEMPTS):
            if stop.is_set():
                raise DownloadError("download cancelled")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise DownloadError(f"{name}: oversized partial; preserved")
            if offset == expected:
                if digest(partial, kind) != checksum:
                    raise DownloadError(f"{name}: partial hash mismatch; preserved")
                os.replace(partial, target)
                return name
            try:
                with content_response(client, name, token, offset) as response:
                    check_status(response)
                    if response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                             response.headers.get("Content-Range", ""))
                        if not match:
                            raise DownloadError(f"{name}: missing or malformed Content-Range")
                        start, end, total = map(int, match.groups())
                        if start != offset or total != expected or not start <= end < total:
                            raise DownloadError(f"{name}: Content-Range does not match partial")
                        length = response.headers.get("Content-Length")
                        if length is not None and length != str(end - start + 1):
                            raise DownloadError(f"{name}: Content-Length does not match range")
                        mode, written = "ab", offset
                    elif response.status_code == 200:
                        length = response.headers.get("Content-Length")
                        if length is not None and length != str(expected):
                            raise DownloadError(f"{name}: wrong Content-Length; partial preserved")
                        mode, written = "wb", 0
                        end = expected - 1
                    else:
                        raise DownloadError(f"{name}: unexpected HTTP {response.status_code}")
                    with partial.open(mode) as output:
                        for block in response.iter_content(1024 * 1024):
                            if stop.is_set():
                                raise DownloadError("download cancelled")
                            if written + len(block) > end + 1:
                                raise DownloadError(f"{name}: response exceeds expected size")
                            output.write(block)
                            written += len(block)
                        output.flush()
                        os.fsync(output.fileno())
                if written != expected:
                    raise RetryableError("incomplete response")
                if digest(partial, kind) != checksum:
                    raise DownloadError(f"{name}: hash mismatch; partial preserved")
                os.replace(partial, target)
                return name
            except (requests.RequestException, RetryableError) as error:
                log(f"{name}: retry {attempt + 1}/{MAX_ATTEMPTS} ({type(error).__name__})")
                if attempt + 1 == MAX_ATTEMPTS:
                    raise NetworkExhausted(f"{name}: retries exhausted ({type(error).__name__})") from None
                stop.wait(min(5 * (attempt + 1), 30))


def write_status(root, state, **extra):
    data = dict(schema="uavrgb.openfly_datagen_download.v1", state=state,
                pid=os.getpid(), checked_at=time.time(), **extra)
    temporary = root / "download_status.json.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / "download_status.json")


def download(jobs, root, token, workers):
    stop = threading.Event()
    totals = dict(files=len(jobs), total_bytes=sum(job[1] for job in jobs), workers=workers)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch, job, root, token, stop) for job in jobs}
        try:
            while pending:
                write_status(root, "downloading", completed_files=completed, **totals)
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    name = future.result()
                    completed += 1
                    log(f"{name}: verified ({completed}/{len(jobs)})")
        except BaseException:
            stop.set()
            for future in pending:
                future.cancel()
            raise
    write_status(root, "complete", completed_files=completed, completed_at=time.time(), **totals)


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-file", type=Path, default=Path("/root/.cache/huggingface/token"))
    parser.add_argument("--workers", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    with (args.output / ".download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, "Another downloader holds the output lock.\n")
        while True:
            try:
                write_status(args.output, "checking_manifest")
                token = args.token_file.read_text().strip()
                if not token:
                    raise DownloadError("HF token file is empty")
                jobs = manifest(token)
                log(f"manifest read: {len(jobs)} assets; checking files and resuming")
                download(jobs, args.output, token, args.workers)
                log("complete; all assets verified, archives remain compressed")
                return 0
            except AccessDenied:
                write_status(args.output, "waiting_for_hf_access")
                log(f"file access denied; retry in {POLL_SECONDS}s")
                try:
                    time.sleep(POLL_SECONDS)
                except KeyboardInterrupt:
                    write_status(args.output, "interrupted")
                    return 130
            except KeyboardInterrupt:
                write_status(args.output, "interrupted")
                return 130
            except Exception as error:
                message = str(error) if isinstance(error, DownloadError) else type(error).__name__
                write_status(args.output, "failed", error=message)
                log(f"failed: {message}")
                return 1 if isinstance(error, NetworkExhausted) else 2


if __name__ == "__main__":
    raise SystemExit(main())
