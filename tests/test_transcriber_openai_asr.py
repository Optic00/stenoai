"""Tests for the OpenAI-compatible batch transcription backend."""

import io
import json
import tempfile
import unittest
import urllib.error
import wave
from pathlib import Path
from unittest.mock import patch

import src.transcriber as transcriber_mod
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
                with self.assertRaisesRegex(RuntimeError, "misconfigured endpoint"):
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
            [{"text": "json response", "start": 0.0, "end": 0.0}],
        )
        self.assertEqual(result["detected_language"], "de")

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

    def test_http_error_redacts_bearer_and_sk_tokens(self):
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
        self.assertIn("Bearer [redacted]", message)
        self.assertIn("[redacted]", message)

    def test_url_guard_rejects_remote_http_and_accepts_loopback_or_https(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_path=Path(tmp_dir) / "config.json")
            self.assertFalse(config.set_openai_asr_api_url("http://evil.example/v1"))
            self.assertTrue(config.set_openai_asr_api_url("http://127.0.0.1:9000/v1"))
            self.assertTrue(config.set_openai_asr_api_url("https://safe.example/v1"))

    def test_language_normalization(self):
        normalize = getattr(transcriber_mod, "_normalize_openai_language")
        self.assertEqual(normalize("English"), "en")
        self.assertEqual(normalize("german"), "de")
        self.assertEqual(normalize("en"), "en")
        self.assertEqual(normalize("Swedish"), "Swedish")
        self.assertIsNone(normalize(None))


if __name__ == "__main__":
    unittest.main()
