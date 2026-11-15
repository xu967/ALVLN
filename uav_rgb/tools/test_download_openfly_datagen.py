import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_openfly_datagen as downloader


class DownloadHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b"scene-data\0" * 250000
        self.name = "airsim/test.zip"
        self.job = (self.name, len(self.payload), "sha256",
                    hashlib.sha256(self.payload).hexdigest())
        self.partial = self.root / (self.name + ".part")
        self.partial.parent.mkdir()
        self.target = self.root / self.name
        self.mode = "resume"
        self.requests = []
        self.content_requests = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append((self.path, self.headers.get("Authorization"),
                                       self.headers.get("Range")))
                if self.path.startswith("/resolve/"):
                    self.send_response(302)
                    self.send_header("Location", owner.cdn + "/content")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                owner.content_requests += 1
                if owner.mode == "denied" or (owner.mode == "busy" and owner.content_requests == 1):
                    self.send_response(403 if owner.mode == "denied" else 503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = int(self.headers.get("Range", "bytes=0-")[6:-1])
                if owner.mode == "ignore_range":
                    start = 0
                body = owner.payload[start:]
                partial = owner.mode != "ignore_range" and "Range" in self.headers
                self.send_response(206 if partial else 200)
                if partial:
                    reported_start = start + 1 if owner.mode == "bad_range" else start
                    self.send_header("Content-Range",
                                     f"bytes {reported_start}-{len(owner.payload) - 1}/{len(owner.payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if owner.mode == "disconnect" and owner.content_requests == 1:
                    self.wfile.write(body[:1024 * 1024 + 32])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 坏 Range 测试会在读取 body 前主动关闭连接。

        def serve():
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def close():
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.addCleanup(close)
            return f"http://127.0.0.1:{server.server_port}"

        self.cdn = serve()
        endpoint = serve()
        self.base_patch = patch.object(downloader, "BASE", endpoint + "/resolve/")
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.stop = threading.Event()

    def fetch(self):
        return downloader.fetch(self.job, self.root, "test-token", self.stop)

    def test_resume_does_not_send_token_to_cdn_or_use_proxy(self):
        self.partial.write_bytes(self.payload[:12345])
        with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1",
                                       "HTTPS_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}):
            self.assertEqual(self.fetch(), self.name)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())
        self.assertEqual(self.requests[0][1], "Bearer test-token")
        self.assertEqual(self.requests[1], ("/content", None, "bytes=12345-"))

    def test_full_partial_is_verified_without_network(self):
        self.partial.write_bytes(self.payload)
        self.fetch()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_bad_full_partial_is_preserved(self):
        corrupted = b"x" * len(self.payload)
        self.partial.write_bytes(corrupted)
        with self.assertRaisesRegex(downloader.DownloadError, "hash mismatch"):
            self.fetch()
        self.assertEqual(self.partial.read_bytes(), corrupted)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.requests, [])

    def test_ignored_range_restarts_without_appending(self):
        self.mode = "ignore_range"
        self.partial.write_bytes(self.payload[:12345])
        self.fetch()
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_wrong_range_does_not_touch_partial(self):
        self.mode = "bad_range"
        before = self.payload[:12345]
        self.partial.write_bytes(before)
        with self.assertRaisesRegex(downloader.DownloadError, "Content-Range"):
            self.fetch()
        self.assertEqual(self.partial.read_bytes(), before)

    def test_interrupted_transfer_resumes_at_saved_offset(self):
        self.mode = "disconnect"
        with patch.object(self.stop, "wait", return_value=False):
            self.fetch()
        self.assertEqual(self.target.read_bytes(), self.payload)
        content = [row for row in self.requests if row[0] == "/content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[1][2], "bytes=1048576-")

    def test_server_error_retries_but_denied_access_does_not(self):
        self.mode = "busy"
        with patch.object(self.stop, "wait", return_value=False):
            self.fetch()
        self.assertEqual(self.content_requests, 2)
        self.target.unlink()
        self.mode = "denied"
        with self.assertRaises(downloader.AccessDenied):
            self.fetch()
        self.assertEqual(self.content_requests, 3)

    def test_retry_limit_preserves_partial(self):
        self.partial.write_bytes(self.payload[:12345])
        with patch.object(downloader, "MAX_ATTEMPTS", 2), \
                patch.object(downloader, "content_response", side_effect=downloader.RetryableError("HTTP 503")), \
                patch.object(self.stop, "wait", return_value=False):
            with self.assertRaisesRegex(downloader.NetworkExhausted, "retries exhausted"):
                self.fetch()
        self.assertEqual(self.partial.read_bytes(), self.payload[:12345])

    def test_complete_status_requires_all_verified_files(self):
        downloader.download([self.job], self.root, "test-token", workers=2)
        status = json.loads((self.root / "download_status.json").read_text())
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["completed_files"], 1)
        self.assertEqual(status["pid"], os.getpid())
        self.assertGreater(status["checked_at"], 0)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_duplicate_process_is_rejected_before_reading_token(self):
        with (self.root / ".download.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(
                [sys.executable, downloader.__file__, "--output", str(self.root),
                 "--token-file", str(self.root / "missing-token")],
                capture_output=True, text=True, timeout=10,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("holds the output lock", result.stderr)
        self.assertFalse((self.root / "download_status.json").exists())


class ManifestTests(unittest.TestCase):
    def test_main_distinguishes_network_and_permanent_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_file = root / "token"
            token_file.write_text("test-token")
            argv = ["download_openfly_datagen.py", "--output", str(root),
                    "--token-file", str(token_file)]
            cases = [(downloader.NetworkExhausted("network retries exhausted"), 1),
                     (downloader.DownloadError("hash mismatch; partial preserved"), 2)]
            for error, expected in cases:
                with self.subTest(error=type(error).__name__), patch.object(sys, "argv", argv), \
                        patch.object(downloader, "manifest", side_effect=error), \
                        patch.object(downloader.signal, "signal"):
                    self.assertEqual(downloader.main(), expected)
                status = json.loads((root / "download_status.json").read_text())
                self.assertEqual(status["state"], "failed")
            token_file.unlink()
            with patch.object(sys, "argv", argv), patch.object(downloader.signal, "signal"):
                self.assertEqual(downloader.main(), 2)

    def test_rejects_unsafe_paths_and_duplicate_entries(self):
        valid = dict(type="file", path="airsim/scene.zip", size=12, lfs={"oid": "a" * 64})
        self.assertEqual(len(downloader.parse_manifest([valid])), 1)
        for path in ("../scene.zip", "airsim/../scene.zip", "/airsim/scene.zip", "airsim//scene.zip"):
            with self.subTest(path=path), self.assertRaises(downloader.DownloadError):
                downloader.parse_manifest([{**valid, "path": path}])
        with self.assertRaises(downloader.DownloadError):
            downloader.parse_manifest([valid, valid])


if __name__ == "__main__":
    unittest.main()
