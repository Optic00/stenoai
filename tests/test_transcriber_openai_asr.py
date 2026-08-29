"""Tests for the OpenAI-compatible batch transcription backend."""

import asyncio
import contextlib
import http.client
import http.server
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import src.config as config_mod
import src.transcriber as transcriber_mod
from simple_recorder import MeetingPipeline, _parse_meeting_markdown
from src.config import Config
from src.transcriber import WhisperTranscriber


OPENAI_ASR_CHUNK_THRESHOLD_BYTES = 24 * 1024 * 1024


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "application/json"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._body

    def close(self):
        pass


class _TricklingResponse(_FakeResponse):
    """A body that never completes until close() proves the wall-clock cap."""
    def __init__(self):
        super().__init__(b"")
        import threading
        self.released = threading.Event()

    def read(self):
        self.released.wait()
        return b""

    def close(self):
        self.released.set()


class _FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        stream = request.data
        self.requests.append(
            {
                "file_size": stream.file_size,
                "prefix": stream.prefix_bytes,
                "timeout": timeout,
            }
        )
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class _BlockingOpener:
    def __init__(self):
        self.entered = threading.Event()
        self.released = threading.Event()

    def open(self, request, timeout):
        self.entered.set()
        self.released.wait()
        return _json_response({"text": "late", "segments": []})


class _DelayedFallbackOpener:
    """Makes each response-format negotiation step consume wall-clock time."""
    def __init__(self, responses, delay_seconds: float):
        self.responses = iter(responses)
        self.delay_seconds = delay_seconds
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        time.sleep(self.delay_seconds)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _json_response(payload):
    return _FakeResponse(json.dumps(payload).encode())


def _http_error(code: int, body: bytes = b"error"):
    return urllib.error.HTTPError(
        "https://api.example/v1/audio/transcriptions",
        code,
        "failure",
        {},
        io.BytesIO(body),
    )


def _streaming_http_error(code: int, response):
    return urllib.error.HTTPError(
        "https://api.example/v1/audio/transcriptions",
        code,
        "failure",
        {},
        response,
    )


def _build_transcriber() -> WhisperTranscriber:
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.model = None
    transcriber.model_size = "large-v3-turbo"
    transcriber.backend = "openai-asr"
    transcriber._openai_asr_api_url = "https://api.example/v1"
    transcriber._openai_asr_api_key = "sk-test-token"
    transcriber._openai_asr_model = "whisper-1"
    return transcriber


def _write_pcm_wav(path: Path, frame_count: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        remaining = frame_count
        silence = b"\0" * (16000 * 2)
        while remaining:
            frames = min(remaining, 16000)
            wav_file.writeframesraw(silence[:frames * 2])
            remaining -= frames


class OpenAiAsrTests(unittest.TestCase):
    def test_loopback_http_upload_bypasses_a_configured_proxy(self):
        target_hits = []
        proxy_hits = []

        class TargetHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                target_hits.append(self.path)
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"text": "local transcript", "segments": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *args):
                pass

        class ProxyHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                proxy_hits.append(self.path)
                self.send_response(502)
                self.end_headers()

            def log_message(self, _format, *args):
                pass

        target = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
        proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        target_thread.start()
        proxy_thread.start()
        try:
            transcriber = _build_transcriber()
            transcriber._openai_asr_api_url = (
                f"http://127.0.0.1:{target.server_port}/v1"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio = Path(tmp_dir) / "short.wav"
                _write_pcm_wav(audio, frame_count=16000)
                with patch(
                    "urllib.request.getproxies",
                    return_value={"http": f"http://127.0.0.1:{proxy.server_port}"},
                ), patch("urllib.request.proxy_bypass", return_value=False):
                    result = transcriber._run_openai_asr(audio, language="en")
            self.assertEqual(result["text"], "local transcript")
            self.assertEqual(len(target_hits), 1)
            self.assertEqual(proxy_hits, [], "API key and audio must never reach HTTP_PROXY")
        finally:
            target.shutdown()
            proxy.shutdown()
            target.server_close()
            proxy.server_close()

    def test_loopback_http_disables_environment_proxies_but_https_keeps_defaults(self):
        endpoints = (
            "http://localhost:9000/v1",
            "http://127.0.0.1:9000/v1",
            "http://[::1]:9000/v1",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint):
                    transcriber = _build_transcriber()
                    transcriber._openai_asr_api_url = endpoint
                    opener = _FakeOpener([_json_response({"text": "local", "segments": []})])
                    with patch("urllib.request.build_opener", return_value=opener) as build:
                        transcriber._run_openai_asr(audio, language="en")
                    proxy_handlers = [
                        handler for handler in build.call_args.args
                        if isinstance(handler, urllib.request.ProxyHandler)
                    ]
                    self.assertEqual(len(proxy_handlers), 1)
                    self.assertEqual(proxy_handlers[0].proxies, {})

            transcriber = _build_transcriber()
            opener = _FakeOpener([_json_response({"text": "remote", "segments": []})])
            with patch("urllib.request.build_opener", return_value=opener) as build:
                transcriber._run_openai_asr(audio, language="en")
            self.assertFalse(any(
                isinstance(handler, urllib.request.ProxyHandler)
                for handler in build.call_args.args
            ), "HTTPS must retain urllib's standard environment-proxy behavior")

    def test_large_wav_is_chunked_and_segment_timestamps_are_offset(self):
        transcriber = _build_transcriber()
        responses = [
            _json_response(
                {
                    "text": "first",
                    "segments": [{"text": "first", "start": 1.0, "end": 2.0}],
                    "duration": 600.0,
                    "language": "english",
                }
            ),
            _json_response(
                {
                    "text": "second",
                    "segments": [{"text": "second", "start": 3.0, "end": 4.0}],
                    "duration": 190.0,
                    "language": "german",
                }
            ),
        ]
        opener = _FakeOpener(responses)
        heartbeats = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "long.wav"
            _write_pcm_wav(audio, frame_count=16000 * 790)
            self.assertGreater(audio.stat().st_size, OPENAI_ASR_CHUNK_THRESHOLD_BYTES)
            with patch("urllib.request.build_opener", return_value=opener), patch(
                "src.transcriber._emit_heartbeat",
                side_effect=lambda done, total: heartbeats.append((done, total)),
            ):
                result = transcriber._run_openai_asr(audio, language="auto")

        self.assertEqual(len(opener.requests), 2)
        self.assertLessEqual(max(req["file_size"] for req in opener.requests), 16000 * 2 * 600 + 44)
        self.assertEqual(result["text"], "first second")
        self.assertEqual(
            result["segments"],
            [
                {"text": "first", "start": 1.0, "end": 2.0},
                {"text": "second", "start": 603.0, "end": 604.0},
            ],
        )
        self.assertEqual(result["duration_seconds"], 790.0)
        self.assertEqual(result["detected_language"], "en")
        self.assertEqual(heartbeats, [(1, 2), (2, 2)])

    def test_file_under_limit_uses_single_request_path(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener(
            [
                _json_response(
                    {
                        "text": "short",
                        "segments": [{"text": "short", "start": 0.5, "end": 1.0}],
                        "duration": 1.0,
                    }
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                result = transcriber._run_openai_asr(audio, language="en")

        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(result["text"], "short")
        self.assertEqual(result["segments"][0]["start"], 0.5)
        self.assertEqual(result["detected_language"], "en")

    def test_request_wait_is_guarded_by_a_heartbeat_context(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener([_json_response({"text": "short", "segments": []})])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener), patch(
                "src.transcriber._heartbeat_while_waiting",
                return_value=contextlib.nullcontext(),
            ) as heartbeat:
                transcriber._run_openai_asr(audio, language="en")

        heartbeat.assert_called_once_with("openai-asr-request")

    def test_total_deadline_closes_a_trickling_response(self):
        response = _TricklingResponse()
        with self.assertRaisesRegex(TimeoutError, "total deadline"):
            transcriber_mod._read_openai_asr_response_with_deadline(
                response, time.monotonic() + 0.01
            )
        self.assertTrue(response.released.is_set())

    def test_run_enforces_deadline_while_opening_request(self):
        transcriber = _build_transcriber()
        opener = _BlockingOpener()
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio = Path(tmp_dir) / "short.wav"
                _write_pcm_wav(audio, frame_count=16000)
                with patch("urllib.request.build_opener", return_value=opener), patch(
                    "src.transcriber.OPENAI_ASR_REQUEST_DEADLINE_SECONDS", 0.02,
                ), patch(
                    "src.transcriber._heartbeat_while_waiting",
                    return_value=contextlib.nullcontext(),
                ):
                    with self.assertRaisesRegex(TimeoutError, "total deadline"):
                        transcriber._run_openai_asr(audio, language="en")
        finally:
            opener.released.set()

        self.assertTrue(opener.entered.is_set())
        self.assertLess(time.monotonic() - started, 1.0)

    def test_run_enforces_deadline_while_reading_success_body(self):
        transcriber = _build_transcriber()
        response = _TricklingResponse()
        opener = _FakeOpener([response])
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener), patch(
                "src.transcriber.OPENAI_ASR_REQUEST_DEADLINE_SECONDS", 0.02,
            ), patch(
                "src.transcriber._heartbeat_while_waiting",
                return_value=contextlib.nullcontext(),
            ):
                with self.assertRaisesRegex(TimeoutError, "total deadline"):
                    transcriber._run_openai_asr(audio, language="en")

        self.assertTrue(response.released.is_set())
        self.assertLess(time.monotonic() - started, 1.0)

    def test_run_never_reads_http_error_body_and_closes_it(self):
        transcriber = _build_transcriber()
        response = _TricklingResponse()
        opener = _FakeOpener([_streaming_http_error(401, response)])
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener), patch(
                "src.transcriber.OPENAI_ASR_REQUEST_DEADLINE_SECONDS", 0.02,
            ), patch(
                "src.transcriber._heartbeat_while_waiting",
                return_value=contextlib.nullcontext(),
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                    transcriber._run_openai_asr(audio, language="en")

        self.assertTrue(response.released.is_set())
        self.assertLess(time.monotonic() - started, 1.0)

    def test_total_deadline_spans_all_response_format_fallbacks(self):
        """Late 400 fallbacks must not reset the one upload attempt's budget."""
        transcriber = _build_transcriber()
        opener = _DelayedFallbackOpener(
            [
                _http_error(400, b"verbose_json unsupported"),
                _http_error(400, b"json unsupported"),
                _FakeResponse(b"late third-format success", "text/plain"),
            ],
            delay_seconds=0.015,
        )
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener), patch(
                "src.transcriber.OPENAI_ASR_REQUEST_DEADLINE_SECONDS", 0.02,
            ), patch(
                "src.transcriber._heartbeat_while_waiting",
                return_value=contextlib.nullcontext(),
            ):
                with self.assertRaisesRegex(TimeoutError, "total deadline"):
                    transcriber._run_openai_asr(audio, language="en")

        self.assertLess(opener.calls, 3, "deadline must prevent the third-format success")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_oversized_non_wav_names_25_mb_limit(self):
        transcriber = _build_transcriber()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "invalid.wav"
            with audio.open("wb") as fh:
                fh.truncate(OPENAI_ASR_CHUNK_THRESHOLD_BYTES + 1)
            with self.assertRaisesRegex(RuntimeError, "25 MB"):
                transcriber._run_openai_asr(audio, language="en")

    def test_verbose_json_missing_text_raises(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener([_json_response({"segments": []})])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaisesRegex(RuntimeError, "text"):
                    transcriber._run_openai_asr(audio, language="auto")

    def test_invalid_200_response_uses_the_transcription_failure_path(self):
        """An invalid success payload is not mistaken for real silence.

        ``transcribe_audio`` is the boundary whose failure flag tells the
        meeting pipeline to retain its recording. Exercise the malformed-200
        response through that boundary, rather than only asserting that the
        lower-level request helper raises.
        """
        transcriber = _build_transcriber()
        opener = _FakeOpener([_json_response({"segments": []})])

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch.object(
                transcriber, "_preprocess_audio", return_value=(audio, False)
            ), patch("urllib.request.build_opener", return_value=opener):
                result = transcriber.transcribe_audio(audio, language="auto")

        self.assertTrue(result["transcription_failed"])
        self.assertIn("containing 'text'", result["error"])

    def test_verbose_json_non_object_raises(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener([_json_response([])])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaisesRegex(RuntimeError, "JSON object"):
                    transcriber._run_openai_asr(audio, language="auto")

    def test_verbose_json_empty_text_requires_empty_segments(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener([_json_response({"text": ""})])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaisesRegex(RuntimeError, "segments"):
                    transcriber._run_openai_asr(audio, language="auto")

    def test_html_content_type_in_text_fallback_is_rejected(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener(
            [
                _FakeResponse(b"not-json"),
                _FakeResponse(b"still-not-json"),
                _FakeResponse(b"proxy landing page", "text/html; charset=utf-8"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaisesRegex(RuntimeError, "unexpected content type"):
                    transcriber._run_openai_asr(audio, language="auto")

    def test_html_body_marker_in_text_fallback_is_rejected(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener(
            [
                _FakeResponse(b"not-json"),
                _FakeResponse(b"still-not-json"),
                _FakeResponse(
                    b"  <!DOCTYPE html><title>Wrong URL</title>",
                    "text/plain",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaisesRegex(RuntimeError, "misconfigured endpoint"):
                    transcriber._run_openai_asr(audio, language="auto")

    def test_text_fallback_accepts_only_text_plain_media_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            problem_failure = None

            transcriber = _build_transcriber()
            opener = _FakeOpener([
                _FakeResponse(b"not-json"),
                _FakeResponse(b"still-not-json"),
                _FakeResponse(b"legitimate transcript", "text/plain; charset=utf-8"),
            ])
            with patch("urllib.request.build_opener", return_value=opener):
                result = transcriber._run_openai_asr(audio, language="en")
            self.assertEqual(result["text"], "legitimate transcript")

            for content_type in (
                "application/problem+json",
                "application/json",
                "application/octet-stream",
                "",
            ):
                with self.subTest(content_type=content_type):
                    transcriber = _build_transcriber()
                    marker = "PRIVATE-PROVIDER-ERROR-BODY"
                    opener = _FakeOpener([
                        _FakeResponse(b"not-json"),
                        _FakeResponse(b"still-not-json"),
                        _FakeResponse(
                            json.dumps({"error": marker}).encode(),
                            content_type,
                        ),
                    ])
                    with patch.object(
                        transcriber, "_preprocess_audio", return_value=(audio, False)
                    ), patch.object(
                        transcriber, "_build_whisper_fallback", return_value=False
                    ), patch("urllib.request.build_opener", return_value=opener):
                        failure = transcriber.transcribe_audio(audio, language="en")
                    self.assertTrue(failure["transcription_failed"])
                    self.assertNotIn(marker, failure["error"])
                    self.assertTrue(audio.exists(), "failed ASR must retain retry audio")
                    if content_type == "application/problem+json":
                        problem_failure = failure

            recorder = MeetingPipeline.__new__(MeetingPipeline)
            recorder.output_dir = Path(tmp_dir) / "output"
            recorder.output_dir.mkdir()
            recorder.transcripts_dir = Path(tmp_dir) / "transcripts"
            recorder.transcripts_dir.mkdir()
            recorder.transcriber = Mock()
            recorder.transcriber.transcribe_diarised.return_value = problem_failure
            recorder.summarizer = None
            config = Mock()
            config.get_language.return_value = "en"
            config.get_whisper_language.return_value = "en"
            config.get_keep_recordings.return_value = False
            with patch("src.config.get_config", return_value=config), patch.dict(
                "os.environ", {"STENOAI_USER_DATA_DIR": tmp_dir}
            ), contextlib.redirect_stdout(io.StringIO()):
                pipeline_result = asyncio.run(
                    recorder.process_recording_streaming(str(audio), "Bad text fallback")
                )
            self.assertTrue(pipeline_result["session_info"]["transcription_failed"])
            self.assertTrue(
                audio.exists(),
                "application/problem+json failure must bypass keep_recordings=false deletion",
            )

    def test_http_error_body_never_reaches_exception_logs_result_or_meeting_metadata(self):
        marker = "PRIVATE-PROVIDER-ERROR-BODY"
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)

            transcriber = _build_transcriber()
            with patch(
                "urllib.request.build_opener",
                return_value=_FakeOpener([_http_error(401, marker.encode())]),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    transcriber._run_openai_asr(audio, language="en")
            self.assertNotIn(marker, str(raised.exception))

            transcriber = _build_transcriber()
            stderr = io.StringIO()
            with self.assertLogs("src.transcriber", level="ERROR") as logs, contextlib.redirect_stderr(stderr), patch.object(
                transcriber, "_preprocess_audio", return_value=(audio, False)
            ), patch.object(
                transcriber, "_build_whisper_fallback", return_value=False
            ), patch(
                "urllib.request.build_opener",
                return_value=_FakeOpener([_http_error(401, marker.encode())]),
            ):
                failure = transcriber.transcribe_audio(audio, language="en")
            self.assertTrue(failure["transcription_failed"])
            self.assertNotIn(marker, failure["error"])
            self.assertNotIn(marker, "\n".join(logs.output))
            self.assertNotIn(marker, stderr.getvalue())

            recorder = MeetingPipeline.__new__(MeetingPipeline)
            recorder.output_dir = Path(tmp_dir) / "output"
            recorder.output_dir.mkdir()
            recorder.transcripts_dir = Path(tmp_dir) / "transcripts"
            recorder.transcripts_dir.mkdir()
            recorder.transcriber = Mock()
            recorder.transcriber.transcribe_diarised.return_value = failure
            recorder.summarizer = None
            config = Mock()
            config.get_language.return_value = "en"
            config.get_whisper_language.return_value = "en"
            config.get_keep_recordings.return_value = False
            stdout = io.StringIO()
            with patch("src.config.get_config", return_value=config), patch.dict(
                "os.environ", {"STENOAI_USER_DATA_DIR": tmp_dir}
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = asyncio.run(
                    recorder.process_recording_streaming(str(audio), "Private failure")
                )

            summary_path = Path(result["session_info"]["summary_file"])
            parsed = _parse_meeting_markdown(summary_path)
            self.assertTrue(audio.exists(), "keep_recordings=false must not delete retry audio")
            self.assertNotIn(marker, stdout.getvalue())
            self.assertNotIn(marker, summary_path.read_text(encoding="utf-8"))
            self.assertNotIn(marker, parsed["session_info"]["error"])

    def test_protocol_error_never_reaches_exception_logs_result_or_meeting_metadata(self):
        marker = "PRIVATE-PROTOCOL-STATUS-LINE"

        def protocol_errors():
            return [
                http.client.BadStatusLine(f"{marker}\r\n"),
                http.client.BadStatusLine(f"{marker}\r\n"),
            ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)

            transcriber = _build_transcriber()
            with patch(
                "urllib.request.build_opener",
                return_value=_FakeOpener(protocol_errors()),
            ), patch("time.sleep"):
                with self.assertRaises(RuntimeError) as raised:
                    transcriber._run_openai_asr(audio, language="en")
            self.assertNotIn(marker, str(raised.exception))

            transcriber = _build_transcriber()
            stderr = io.StringIO()
            with self.assertLogs(
                "src.transcriber", level="ERROR"
            ) as logs, contextlib.redirect_stderr(stderr), patch.object(
                transcriber, "_preprocess_audio", return_value=(audio, False)
            ), patch.object(
                transcriber, "_build_whisper_fallback", return_value=False
            ), patch(
                "urllib.request.build_opener",
                return_value=_FakeOpener(protocol_errors()),
            ), patch("time.sleep"):
                failure = transcriber.transcribe_audio(audio, language="en")
            self.assertTrue(failure["transcription_failed"])
            self.assertNotIn(marker, failure["error"])
            self.assertNotIn(marker, "\n".join(logs.output))
            self.assertNotIn(marker, stderr.getvalue())

            recorder = MeetingPipeline.__new__(MeetingPipeline)
            recorder.output_dir = Path(tmp_dir) / "output"
            recorder.output_dir.mkdir()
            recorder.transcripts_dir = Path(tmp_dir) / "transcripts"
            recorder.transcripts_dir.mkdir()
            recorder.transcriber = Mock()
            recorder.transcriber.transcribe_diarised.return_value = failure
            recorder.summarizer = None
            config = Mock()
            config.get_language.return_value = "en"
            config.get_whisper_language.return_value = "en"
            config.get_keep_recordings.return_value = False
            stdout = io.StringIO()
            with patch("src.config.get_config", return_value=config), patch.dict(
                "os.environ", {"STENOAI_USER_DATA_DIR": tmp_dir}
            ), contextlib.redirect_stdout(stdout):
                result = asyncio.run(
                    recorder.process_recording_streaming(str(audio), "Protocol failure")
                )

            summary_path = Path(result["session_info"]["summary_file"])
            parsed = _parse_meeting_markdown(summary_path)
            self.assertTrue(audio.exists(), "protocol failure must preserve retry audio")
            self.assertNotIn(marker, stdout.getvalue())
            self.assertNotIn(marker, summary_path.read_text(encoding="utf-8"))
            self.assertNotIn(marker, parsed["session_info"]["error"])

    def test_provider_language_is_validated_before_persistence(self):
        marker = "PRIVATE-LANGUAGE-MARKER"
        injected_language = f"en\n---\ntranscription_failed: true\n{marker}"
        payload = {
            "text": "A legitimate transcript with enough words for detection.",
            "segments": [
                {
                    "text": "A legitimate transcript with enough words for detection.",
                    "start": 0,
                    "end": 2,
                }
            ],
            "duration": 2,
            "language": injected_language,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            transcriber = _build_transcriber()
            with patch(
                "urllib.request.build_opener",
                return_value=_FakeOpener([_json_response(payload)]),
            ):
                transcribed = transcriber._run_openai_asr(audio, language="auto")
            self.assertIsNone(transcribed["detected_language"])

            recorder = MeetingPipeline.__new__(MeetingPipeline)
            recorder.output_dir = Path(tmp_dir) / "output"
            recorder.output_dir.mkdir()
            recorder.transcripts_dir = Path(tmp_dir) / "transcripts"
            recorder.transcripts_dir.mkdir()
            recorder.transcriber = Mock()
            recorder.transcriber.transcribe_diarised.return_value = transcribed
            recorder.summarizer = None
            config = Mock()
            config.get_language.return_value = "auto"
            config.get_whisper_language.return_value = "auto"
            config.get_language_name.side_effect = lambda code: code or "Unknown"
            config.get_auto_summarize_enabled.return_value = False
            config.get_keep_recordings.return_value = True
            with patch("src.config.get_config", return_value=config), patch.dict(
                "os.environ", {"STENOAI_USER_DATA_DIR": tmp_dir}
            ), contextlib.redirect_stdout(io.StringIO()):
                result = asyncio.run(
                    recorder.process_recording_streaming(str(audio), "Safe language")
                )

            summary_path = Path(result["session_info"]["summary_file"])
            transcript_path = recorder.transcripts_dir / "short_transcript.txt"
            persisted = summary_path.read_text(encoding="utf-8")
            self.assertNotIn(marker, persisted)
            self.assertNotIn(marker, transcript_path.read_text(encoding="utf-8"))
            self.assertIsNone(
                _parse_meeting_markdown(summary_path)["session_info"]["detected_language"]
            )

    def test_fallback_warning_never_includes_http_error_body(self):
        marker = "PRIVATE-PROVIDER-ERROR-BODY"
        transcriber = _build_transcriber()
        opener = _FakeOpener([
            _http_error(400, marker.encode()),
            _json_response({"text": "safe transcript"}),
        ])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with self.assertLogs("src.transcriber", level="WARNING") as logs, patch(
                "urllib.request.build_opener", return_value=opener
            ):
                result = transcriber._run_openai_asr(audio, language="en")
        self.assertEqual(result["text"], "safe transcript")
        self.assertNotIn(marker, "\n".join(logs.output))

    def test_invalid_provider_numeric_metadata_never_reaches_failure_output(self):
        marker = "PRIVATE-PROVIDER-METADATA"
        payloads = (
            {
                "text": "provider transcript",
                "segments": [{"text": "provider transcript", "start": marker, "end": 1}],
            },
            {
                "text": "provider transcript",
                "segments": [{"text": "provider transcript", "start": 0, "end": 1}],
                "duration": marker,
            },
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            for payload in payloads:
                with self.subTest(payload=payload):
                    transcriber = _build_transcriber()
                    with self.assertLogs("src.transcriber", level="ERROR") as logs, patch.object(
                        transcriber, "_preprocess_audio", return_value=(audio, False)
                    ), patch.object(
                        transcriber, "_build_whisper_fallback", return_value=False
                    ), patch(
                        "urllib.request.build_opener",
                        return_value=_FakeOpener([_json_response(payload)]),
                    ):
                        failure = transcriber.transcribe_audio(audio, language="en")
                    self.assertTrue(failure["transcription_failed"])
                    self.assertNotIn(marker, failure["error"])
                    self.assertNotIn(marker, "\n".join(logs.output))

    def test_json_rung_fallback_synthesizes_segment(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener(
            [
                _http_error(400, b"verbose_json unsupported"),
                _json_response({"text": "json response"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                result = transcriber._run_openai_asr(audio, language="de")

        self.assertEqual(len(opener.requests), 2)
        self.assertIn(b'name="response_format"\r\n\r\njson\r\n', opener.requests[1]["prefix"])
        self.assertEqual(
            result["segments"],
            [{
                "text": "json response", "start": 0.0, "end": 0.0,
                "has_timestamps": False,
            }],
        )
        self.assertEqual(result["detected_language"], "de")

    def test_verbose_json_text_without_segments_is_preserved_as_untimed(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener([_json_response({"text": "whole-channel response"})])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                result = transcriber._run_openai_asr(audio, language="en")

        self.assertEqual(result["text"], "whole-channel response")
        self.assertEqual(result["segments"], [{
            "text": "whole-channel response", "start": 0.0, "end": 0.0,
            "has_timestamps": False,
        }])

    def test_429_is_retried_once_then_succeeds(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener(
            [
                _http_error(429, b"slow down"),
                _json_response({"text": "recovered", "segments": []}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener), patch(
                "time.sleep"
            ) as sleep_mock:
                result = transcriber._run_openai_asr(audio, language="en")

        self.assertEqual(result["text"], "recovered")
        self.assertEqual(len(opener.requests), 2)
        sleep_mock.assert_called_once_with(2)

    def test_http_error_drops_bearer_and_sk_body_entirely(self):
        transcriber = _build_transcriber()
        opener = _FakeOpener(
            [
                _http_error(
                    401,
                    b"Authorization: Bearer secret-token\nprovider key sk-abcdefgh12345678",
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaises(RuntimeError) as raised:
                    transcriber._run_openai_asr(audio, language="en")

        message = str(raised.exception)
        self.assertNotIn("secret-token", message)
        self.assertNotIn("sk-abcdefgh12345678", message)
        self.assertEqual(message, "openai-asr HTTP 401")

    def test_http_error_drops_configured_non_openai_key_body_entirely(self):
        transcriber = _build_transcriber()
        configured_key = "test-provider-credential-987654"
        transcriber._openai_asr_api_key = configured_key
        opener = _FakeOpener([
            _http_error(401, f"token={configured_key} api_key=other-test-token".encode())
        ])
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio = Path(tmp_dir) / "short.wav"
            _write_pcm_wav(audio, frame_count=16000)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaises(RuntimeError) as raised:
                    transcriber._run_openai_asr(audio, language="en")

        message = str(raised.exception)
        self.assertNotIn(configured_key, message)
        self.assertNotIn("other-test-token", message)
        self.assertEqual(message, "openai-asr HTTP 401")

    def test_config_diagnostic_redacts_userinfo_and_query_credentials(self):
        redacted = config_mod._redact_url_credentials(
            "http://user:password@evil.example/v1?access_token=secret-value"
        )
        self.assertEqual(redacted, "[redacted-url]")

    def test_url_guard_rejects_remote_http_and_accepts_loopback_or_https(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_path=Path(tmp_dir) / "config.json")
            self.assertFalse(config.set_openai_asr_api_url("http://evil.example/v1"))
            self.assertTrue(config.set_openai_asr_api_url("http://127.0.0.1:9000/v1"))
            self.assertTrue(config.set_openai_asr_api_url("https://safe.example/v1"))

    def test_legacy_unsafe_url_fails_closed_and_never_reaches_the_transcriber(self):
        unsafe_urls = (
            "http://user:password@evil.example/v1",
            "https://safe.example/v1?key=secret",
            "https://safe.example/v1?subscription-key=secret",
            "https://safe.example/v1?sig=secret",
            "https://safe.example/v1?unknown=secret",
        )
        for unsafe_url in unsafe_urls:
            with self.subTest(url=unsafe_url), tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.json"
                config_path.write_text(json.dumps({"openai_asr_api_url": unsafe_url}))
                config = Config(config_path=config_path)
                self.assertEqual(config.get_openai_asr_api_url(), "")

                transcriber = _build_transcriber()
                transcriber._openai_asr_api_url = unsafe_url
                with patch("urllib.request.build_opener") as build_opener:
                    with self.assertRaisesRegex(RuntimeError, "unsafe or invalid"):
                        transcriber._run_openai_asr(Path("unused.wav"), language="en")
                build_opener.assert_not_called()

    def test_language_normalization(self):
        normalize = getattr(transcriber_mod, "_normalize_openai_language")
        self.assertEqual(normalize("English"), "en")
        self.assertEqual(normalize("german"), "de")
        self.assertEqual(normalize("en"), "en")
        self.assertEqual(normalize(" EN "), "en")
        self.assertEqual(normalize("haw"), "haw")
        self.assertEqual(normalize("uk"), "uk")
        self.assertEqual(normalize("Finnish"), "fi")
        self.assertEqual(normalize("Swedish"), "sv")
        self.assertIsNone(normalize("xx"))
        self.assertIsNone(normalize("zzz"))
        self.assertIsNone(normalize("unknown language"))
        self.assertIsNone(normalize("en\n---\ninjected: true"))
        self.assertIsNone(normalize(42))
        self.assertIsNone(normalize(None))


if __name__ == "__main__":
    unittest.main()
