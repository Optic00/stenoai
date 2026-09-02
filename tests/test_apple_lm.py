"""Unit tests for Apple SystemLanguageModel integration."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from click.testing import CliRunner

import simple_recorder
from src.apple_lm import (
    APPLE_SYSTEM_MODEL,
    APPLE_LM_NUM_CTX,
    AppleLMClient,
    _helper_app_for_binary,
    _run_apple_lm_app,
    apple_lm_generation_error_message,
    apple_lm_status,
    apple_lm_should_list,
    apple_system_model_info,
    apple_lm_unavailable_message,
    complete,
    is_apple_system_model,
    resolve_apple_lm_bin,
    reset_apple_lm_cache,
    stream_complete,
)
from src.config import Config
from src.summarizer import OllamaSummarizer, resolve_num_ctx


class BaseAppleLMTest(unittest.TestCase):
    def setUp(self):
        reset_apple_lm_cache()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._old_user_data = os.environ.get("STENOAI_USER_DATA_DIR")
        os.environ["STENOAI_USER_DATA_DIR"] = self._tmp_dir.name

    def tearDown(self):
        reset_apple_lm_cache()
        if self._old_user_data is not None:
            os.environ["STENOAI_USER_DATA_DIR"] = self._old_user_data
        else:
            os.environ.pop("STENOAI_USER_DATA_DIR", None)
        self._tmp_dir.cleanup()


class AppleLMResolutionTests(BaseAppleLMTest):
    def test_is_apple_system_model(self):
        self.assertTrue(is_apple_system_model("apple:system"))
        self.assertFalse(is_apple_system_model("gemma4:e2b-it-qat"))
        self.assertFalse(is_apple_system_model(None))

    def test_helper_app_is_recognized_from_canonical_executable(self):
        binary = (
            Path("/Applications/Steno.app/Contents/Helpers")
            / "Steno Apple LM.app"
            / "Contents"
            / "MacOS"
            / "steno-apple-lm"
        )

        self.assertEqual(
            _helper_app_for_binary(str(binary)),
            Path("/Applications/Steno.app/Contents/Helpers/Steno Apple LM.app"),
        )
        self.assertIsNone(_helper_app_for_binary("/tmp/mock-steno-apple-lm"))

    def test_frozen_backend_resolves_nested_helper(self):
        executable = (
            "/Applications/Steno.app/Contents/Helpers/Steno Apple LM.app/"
            "Contents/MacOS/steno-apple-lm"
        )
        with mock.patch("src.apple_lm.sys.platform", "darwin"), mock.patch.object(
            sys,
            "frozen",
            True,
            create=True,
        ), mock.patch(
            "src.apple_lm.sys.executable",
            "/Applications/Steno.app/Contents/Resources/stenoai/stenoai",
        ), mock.patch(
            "src.apple_lm.os.access",
            side_effect=lambda path, _mode: str(path) == executable,
        ), mock.patch.dict(
            os.environ,
            {"STENOAI_DISABLE_APPLE_LM": "0"},
        ):
            self.assertEqual(resolve_apple_lm_bin(), executable)

    def test_relative_override_is_canonicalized(self):
        override = Path(self._tmp_dir.name) / "mock-steno-apple-lm"
        override.write_bytes(b"mock")
        override.chmod(0o755)
        relative = os.path.relpath(override, Path.cwd())

        with mock.patch("src.apple_lm.sys.platform", "darwin"), mock.patch.dict(
            os.environ,
            {
                "STENOAI_APPLE_LM_BIN": relative,
                "STENOAI_DISABLE_APPLE_LM": "0",
            },
        ):
            self.assertEqual(resolve_apple_lm_bin(), str(override.resolve()))

    def test_unavailable_message_maps_fixed_reason(self):
        self.assertIn(
            "still downloading",
            apple_lm_unavailable_message({"available": False, "reason": "modelNotReady"}),
        )

    def test_pre_tahoe_status_does_not_start_sidecar(self):
        with mock.patch("src.apple_lm.sys.platform", "darwin"), \
             mock.patch("src.apple_lm.platform.mac_ver", return_value=("15.7", ("", "", ""), "")), \
             mock.patch.dict(os.environ, {"STENOAI_DISABLE_APPLE_LM": "0"}), \
             mock.patch("src.apple_lm._run_apple_lm") as run_sidecar:
            from src.apple_lm import apple_lm_status

            self.assertEqual(
                apple_lm_status(),
                {"available": False, "reason": "unsupported_os"},
            )
        run_sidecar.assert_not_called()

    def test_e2e_mock_sidecar_can_run_on_pre_tahoe_runner(self):
        env = {
            "STENOAI_DISABLE_APPLE_LM": "0",
            "STENOAI_E2E": "1",
            "STENOAI_APPLE_LM_BIN": "/tmp/mock-apple-lm",
        }
        with mock.patch("src.apple_lm.sys.platform", "darwin"), \
             mock.patch("src.apple_lm.platform.mac_ver", return_value=("15.7", ("", "", ""), "")), \
             mock.patch.dict(os.environ, env, clear=False), \
             mock.patch("src.apple_lm.resolve_apple_lm_bin", return_value="/tmp/mock-apple-lm"), \
             mock.patch(
                 "src.apple_lm._run_apple_lm",
                 return_value=json.dumps({"available": True}),
             ) as run_sidecar:
            from src.apple_lm import apple_lm_status

            self.assertEqual(apple_lm_status(), {"available": True})
        run_sidecar.assert_called_once_with(["status"], timeout=15)

    def test_transient_status_failure_is_not_cached(self):
        env = {"STENOAI_DISABLE_APPLE_LM": "0"}
        with mock.patch("src.apple_lm.sys.platform", "darwin"), mock.patch(
            "src.apple_lm.platform.mac_ver",
            return_value=("27.0", ("", "", ""), ""),
        ), mock.patch.dict(os.environ, env, clear=False), mock.patch(
            "src.apple_lm.resolve_apple_lm_bin",
            return_value="/tmp/steno-apple-lm",
        ), mock.patch(
            "src.apple_lm._run_apple_lm",
            side_effect=[
                RuntimeError("temporary launch failure"),
                json.dumps({"available": True}),
            ],
        ) as run_sidecar:
            self.assertEqual(
                apple_lm_status(),
                {"available": False, "reason": "sidecar_error"},
            )
            self.assertEqual(apple_lm_status(), {"available": True})

        self.assertEqual(run_sidecar.call_count, 2)

    def test_apple_system_model_info_describes_os_managed_model(self):
        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True, "display_name": "Apple Intelligence"}):
            info = apple_system_model_info(is_default=True)
            self.assertIn("OS-managed", info["description"])
            self.assertIn("(selected)", info["description"])

    def test_actionable_unavailability_is_listed(self):
        status = {"available": False, "reason": "appleIntelligenceNotEnabled"}
        self.assertTrue(apple_lm_should_list(status))
        self.assertFalse(
            apple_lm_should_list({"available": False, "reason": "unsupported_os"})
        )

    def test_generation_error_maps_fixed_reason(self):
        self.assertIn("declined", apple_lm_generation_error_message("refusal"))
        self.assertEqual(
            apple_lm_generation_error_message("unknown"),
            "Apple Intelligence request failed",
        )

    def test_complete_preserves_fixed_refusal_reason(self):
        payload = json.dumps({"error": "apple_lm_failed", "reason": "refusal"})
        with mock.patch("src.apple_lm._run_apple_lm", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "declined"):
                complete("synthetic prompt")

    def test_launch_services_preserves_helper_error_payload(self):
        payload = json.dumps(
            {"error": "apple_lm_failed", "reason": "guardrail"}
        )
        invocation = mock.Mock()
        invocation.iter_lines.return_value = iter([payload])
        invocation.wait.side_effect = RuntimeError("launcher failed")

        with mock.patch(
            "src.apple_lm._AppleLMAppInvocation",
            return_value=invocation,
        ):
            self.assertEqual(
                _run_apple_lm_app(
                    Path("/tmp/Steno Apple LM.app"),
                    ["complete"],
                    stdin="synthetic prompt",
                    timeout=30,
                ),
                payload,
            )

        invocation.close.assert_called_once_with()

    def test_launch_services_preserves_launcher_failure(self):
        invocation = mock.Mock()
        invocation.iter_lines.return_value = iter([])
        invocation.wait.side_effect = RuntimeError("launcher failed")

        with mock.patch(
            "src.apple_lm._AppleLMAppInvocation",
            return_value=invocation,
        ):
            with self.assertRaisesRegex(RuntimeError, "launcher failed"):
                _run_apple_lm_app(
                    Path("/tmp/Steno Apple LM.app"),
                    ["status"],
                    stdin=None,
                    timeout=30,
                )

        invocation.close.assert_called_once_with()

    def test_complete_uses_launch_services_for_helper_app(self):
        app = Path(self._tmp_dir.name) / "Steno Apple LM.app"
        binary = app / "Contents" / "MacOS" / "steno-apple-lm"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"helper")
        binary.chmod(0o755)

        with mock.patch(
            "src.apple_lm.resolve_apple_lm_bin",
            return_value=str(binary),
        ), mock.patch(
            "src.apple_lm._run_apple_lm_app",
            return_value=json.dumps({"text": "Sandboxed response"}),
        ) as run_app, mock.patch("src.apple_lm.subprocess.run") as run_direct:
            self.assertEqual(complete("synthetic prompt"), "Sandboxed response")

        run_app.assert_called_once_with(
            app,
            ["complete"],
            stdin="synthetic prompt",
            timeout=7200,
        )
        run_direct.assert_not_called()

    def test_nonzero_sidecar_preserves_fixed_guardrail_reason(self):
        failed = subprocess.CompletedProcess(
            args=["steno-apple-lm", "complete"],
            returncode=1,
            stdout=json.dumps(
                {"error": "apple_lm_failed", "reason": "guardrail"}
            ),
            stderr="",
        )
        with mock.patch("src.apple_lm.resolve_apple_lm_bin", return_value="sidecar"), \
             mock.patch("src.apple_lm.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "could not process"):
                complete("synthetic prompt")

    @unittest.skipIf(sys.platform == "win32", "POSIX executable fixture")
    def test_stream_timeout_terminates_sidecar_process(self):
        script = Path(self._tmp_dir.name) / "slow-apple-lm"
        script.write_text(
            f"#!{sys.executable}\n"
            "import sys, time\n"
            "sys.stdin.read()\n"
            "time.sleep(30)\n"
        )
        script.chmod(0o755)
        captured = []
        real_popen = subprocess.Popen

        def capture_process(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            captured.append(proc)
            return proc

        with mock.patch("src.apple_lm.resolve_apple_lm_bin", return_value=str(script)), \
             mock.patch("src.apple_lm.subprocess.Popen", side_effect=capture_process):
            with self.assertRaisesRegex(TimeoutError, "stream timed out"):
                list(stream_complete("synthetic prompt", timeout=0.05))

        self.assertEqual(len(captured), 1)
        self.assertIsNotNone(captured[0].poll())

    def test_stream_uses_launch_services_for_helper_app(self):
        app = Path(self._tmp_dir.name) / "Steno Apple LM.app"
        binary = app / "Contents" / "MacOS" / "steno-apple-lm"

        with mock.patch(
            "src.apple_lm.resolve_apple_lm_bin",
            return_value=str(binary),
        ), mock.patch(
            "src.apple_lm._stream_apple_lm_app",
            return_value=iter(["Hello", " world"]),
        ) as stream_app, mock.patch("src.apple_lm.subprocess.Popen") as popen:
            self.assertEqual(
                list(stream_complete("synthetic prompt")),
                ["Hello", " world"],
            )

        stream_app.assert_called_once_with(app, "synthetic prompt", 7200)
        popen.assert_not_called()


class AppleLMConfigOptInTests(BaseAppleLMTest):
    def test_fresh_config_does_not_probe_or_adopt_apple_system(self):
        cfg_path = Path(self._tmp_dir.name) / "config.json"
        with mock.patch("src.apple_lm.apple_lm_available") as available, \
             mock.patch("src.apple_lm.apple_lm_status") as status:
            config = Config(config_path=cfg_path)
        self.assertEqual(config.get_model(), Config.DEFAULT_MODEL)
        available.assert_not_called()
        status.assert_not_called()

    def test_existing_auto_config_is_not_changed_when_apple_is_available(self):
        cfg_path = Path(self._tmp_dir.name) / "config.json"
        cfg_path.write_text(json.dumps({"model": Config.DEFAULT_MODEL, "summary_model_source": "auto"}))
        with mock.patch("src.apple_lm.apple_lm_available", return_value=True):
            config = Config(config_path=cfg_path)
        self.assertEqual(config.get_model(), Config.DEFAULT_MODEL)

    def test_explicit_apple_choice_survives_temporary_unavailability(self):
        cfg_path = Path(self._tmp_dir.name) / "config.json"
        cfg_path.write_text(json.dumps({"model": APPLE_SYSTEM_MODEL, "summary_model_source": "user"}))
        with mock.patch("src.apple_lm.apple_lm_available", return_value=False):
            config = Config(config_path=cfg_path)
        self.assertEqual(config.get_model(), APPLE_SYSTEM_MODEL)

    def test_explicit_user_choice_is_not_overwritten_by_auto_adoption(self):
        cfg_path = Path(self._tmp_dir.name) / "config.json"
        cfg_path.write_text(json.dumps({"model": "qwen3.5:9b", "summary_model_source": "user"}))
        with mock.patch("src.apple_lm.apple_lm_available", return_value=True):
            config = Config(config_path=cfg_path)
            self.assertEqual(config.get_model(), "qwen3.5:9b")

    def test_get_model_info_returns_apple_metadata(self):
        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            config = Config(config_path=Path(self._tmp_dir.name) / "config.json")
            info = config.get_model_info(APPLE_SYSTEM_MODEL)
            self.assertIsNotNone(info)
            self.assertEqual(info["name"], "Apple Intelligence")
            self.assertEqual(info["params"], "OS-managed")


class AppleLMSummarizerIntegrationTests(BaseAppleLMTest):
    def test_resolve_num_ctx_for_apple_system(self):
        self.assertEqual(resolve_num_ctx(APPLE_SYSTEM_MODEL), APPLE_LM_NUM_CTX)

    def test_summarizer_initializes_apple_client_without_ollama(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = APPLE_SYSTEM_MODEL

        with mock.patch("src.summarizer.OLLAMA_AVAILABLE", False), \
             mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            summarizer = OllamaSummarizer(config=cfg)
            self.assertTrue(summarizer._using_apple_lm())
            self.assertIsInstance(summarizer.client, AppleLMClient)
            self.assertTrue(summarizer._ensure_ollama_ready())
    def test_apple_lm_client_chat(self):
        client = AppleLMClient()
        with mock.patch("src.apple_lm.complete", return_value="Summary of meeting"):
            res = client.chat(messages=[{"role": "user", "content": "summarize"}])
            self.assertEqual(res, {"message": {"content": "Summary of meeting"}})

    def test_apple_lm_client_stream(self):
        client = AppleLMClient()
        with mock.patch("src.apple_lm.stream_complete", return_value=iter(["Hello", " world"])):
            stream = client.chat(stream=True, messages=[{"role": "user", "content": "hi"}])
            chunks = [c["message"]["content"] for c in stream]
            self.assertEqual(chunks, ["Hello", " world"])

    def test_generate_title_routes_through_apple_lm(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = APPLE_SYSTEM_MODEL
        cfg.get_language_name.return_value = "English"

        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            summarizer = OllamaSummarizer(config=cfg)
            with mock.patch("src.apple_lm.complete", return_value="Project Kickoff"):
                title = summarizer.generate_title("Summary here", "transcript")
                self.assertEqual(title, "Project Kickoff")

    def test_summarizer_fails_visibly_when_apple_configured_but_unavailable(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = APPLE_SYSTEM_MODEL
        cfg.DEFAULT_MODEL = "gemma4:e2b-it-qat"

        status = {"available": False, "reason": "appleIntelligenceNotEnabled"}
        with mock.patch("src.apple_lm.apple_lm_status", return_value=status), \
             mock.patch.object(OllamaSummarizer, "_ensure_ollama_ready") as ensure, \
             mock.patch("ollama.Client") as ollama_client:
            with self.assertRaisesRegex(RuntimeError, "Enable Apple Intelligence"):
                OllamaSummarizer(config=cfg)
        ensure.assert_not_called()
        ollama_client.assert_not_called()

    def test_query_transcript_ollama_retry_loop_defines_max_retries(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = "gemma4:e2b-it-qat"
        cfg.DEFAULT_MODEL = "gemma4:e2b-it-qat"

        with mock.patch.object(OllamaSummarizer, "_ensure_ollama_ready"), \
             mock.patch("ollama.Client"):
            summarizer = OllamaSummarizer(config=cfg)
            summarizer.client = mock.MagicMock()
            summarizer.client.chat.return_value = {"message": {"content": "Answer here"}}
            res = summarizer.query_transcript("Transcript", "Question?")
            self.assertEqual(res, "Answer here")
            summarizer.client.chat.assert_called_once()

    def test_generate_title_passes_timeout_to_complete(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = APPLE_SYSTEM_MODEL
        cfg.get_language_name.return_value = "English"

        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            summarizer = OllamaSummarizer(config=cfg)
            with mock.patch("src.apple_lm.complete", return_value="Project Title") as mock_complete:
                title = summarizer.generate_title("Summary here", "transcript")
                self.assertEqual(title, "Project Title")
                mock_complete.assert_called_once()
                self.assertEqual(mock_complete.call_args.kwargs.get("timeout"), 90)

    def test_long_transcript_uses_snapshot_compact(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = APPLE_SYSTEM_MODEL
        cfg.get_language_name.return_value = "English"
        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            summarizer = OllamaSummarizer(config=cfg)
        prompts = []

        def fake_complete(prompt, timeout=7200):
            prompts.append(prompt)
            return f"SNAPSHOT-{len(prompts)}\nDECISIONS\n- keep going"

        def fake_stream(prompt, timeout=7200):
            yield "## Summary\nSnapshot formatted."

        transcript = "".join(
            f"Speaker A: unique-line-{i} was discussed.\n" for i in range(500)
        )
        with mock.patch("src.apple_lm.complete", side_effect=fake_complete), \
             mock.patch("src.apple_lm.stream_complete", side_effect=fake_stream), \
             mock.patch.object(summarizer, "_map_reduce_streaming") as map_reduce:
            text = "".join(summarizer.summarize_transcript_streaming(transcript))
        map_reduce.assert_not_called()
        self.assertIn("Snapshot formatted.", text)
        self.assertGreaterEqual(len(prompts), 2)
        self.assertTrue(all("CURRENT SNAPSHOT" in p for p in prompts))
        self.assertIn("SNAPSHOT-1", prompts[1])
        joined = "\n".join(prompts)
        self.assertIn("unique-line-0", joined)
        self.assertIn("unique-line-499", joined)

    def test_hard_trim_snapshot_keeps_head_and_tail(self):
        cfg = mock.Mock()
        cfg.get_ai_provider.return_value = "local"
        cfg.get_remote_ollama_url.return_value = None
        cfg.get_model.return_value = APPLE_SYSTEM_MODEL
        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            summarizer = OllamaSummarizer(config=cfg)
        from src.summarizer import _SNAPSHOT_MAX_CHARS
        blob = "H" * 2000 + "MID" + "T" * 2000
        trimmed = summarizer._hard_trim_snapshot(blob)
        self.assertLessEqual(len(trimmed), _SNAPSHOT_MAX_CHARS)
        self.assertTrue(trimmed.startswith("H"))
        self.assertTrue(trimmed.endswith("T"))
        self.assertIn("...", trimmed)


class AppleLMCLITests(BaseAppleLMTest):
    def test_list_models_prepends_apple_system_when_available(self):
        with mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}), \
             mock.patch("src.config.Config.get_model", return_value=APPLE_SYSTEM_MODEL):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.list_models)
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertIn(APPLE_SYSTEM_MODEL, data["supported_models"])
            self.assertTrue(data["supported_models"][APPLE_SYSTEM_MODEL]["installed"])

    def test_list_models_shows_apple_system_not_installed_when_unavailable(self):
        status = {"available": False, "reason": "appleIntelligenceNotEnabled"}
        with mock.patch("src.apple_lm.apple_lm_status", return_value=status), \
             mock.patch("src.config.Config.get_model", return_value=APPLE_SYSTEM_MODEL):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.list_models)
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertIn(APPLE_SYSTEM_MODEL, data["supported_models"])
            self.assertFalse(data["supported_models"][APPLE_SYSTEM_MODEL]["installed"])

    def test_list_models_offers_actionable_unavailable_apple_system(self):
        status = {"available": False, "reason": "appleIntelligenceNotEnabled"}
        with mock.patch("src.apple_lm.apple_lm_status", return_value=status), \
             mock.patch("src.config.Config.get_model", return_value=Config.DEFAULT_MODEL):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.list_models)
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            apple = data["supported_models"][APPLE_SYSTEM_MODEL]
            self.assertFalse(apple["installed"])
            self.assertIn("Enable Apple Intelligence", apple["description"])

    def test_list_models_hides_apple_system_on_unsupported_os(self):
        status = {"available": False, "reason": "unsupported_os"}
        with mock.patch("src.apple_lm.apple_lm_status", return_value=status), \
             mock.patch("src.config.Config.get_model", return_value=Config.DEFAULT_MODEL):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.list_models)
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertNotIn(APPLE_SYSTEM_MODEL, data["supported_models"])

    def test_check_model_for_apple_system(self):
        with mock.patch("src.apple_lm.apple_lm_available", return_value=True):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.check_model, [APPLE_SYSTEM_MODEL])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertTrue(data["installed"])

    def test_pull_model_for_apple_system_succeeds_when_available(self):
        with mock.patch("src.apple_lm.apple_lm_available", return_value=True):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.pull_model, [APPLE_SYSTEM_MODEL])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertTrue(data["success"])

    def test_delete_model_refuses_apple_system(self):
        runner = CliRunner()
        result = runner.invoke(simple_recorder.delete_model, [APPLE_SYSTEM_MODEL])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertFalse(data["success"])
        self.assertIn("Cannot delete", data["error"])

    def test_resolve_setup_model_does_not_auto_select_apple_system(self):
        with mock.patch("src.apple_lm.apple_lm_available", return_value=True), \
             mock.patch("src.ollama_manager.start_ollama_server"), \
             mock.patch("ollama.list", return_value=mock.Mock(models=[])):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.resolve_setup_model)
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertIsNone(data["installed"])
            self.assertIsNotNone(data["pull_target"])

    def test_setup_check_reports_apple_system_model(self):
        with mock.patch("sys.platform", "darwin"), \
             mock.patch("src.config.Config.get_model", return_value=APPLE_SYSTEM_MODEL), \
             mock.patch("src.apple_lm.apple_lm_status", return_value={"available": True}):
            runner = CliRunner()
            result = runner.invoke(simple_recorder.setup_check, ["--json"])
            self.assertEqual(result.exit_code, 0)
            lines = [line for line in result.output.splitlines() if line.strip().startswith("{")]
            data = json.loads(lines[0])
            llm_check = next((c for c in data["checks"] if c["name"] == "llm-model"), None)
            self.assertIsNotNone(llm_check)
            self.assertEqual(llm_check["status"], "pass")
            self.assertIn("Apple System Language Model (available)", llm_check["detail"])


if __name__ == "__main__":
    unittest.main()
