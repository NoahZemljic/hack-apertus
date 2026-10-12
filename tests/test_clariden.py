"""GPU-free deployment tests: isolation, cache safety, readiness and lifecycle."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from containers.install_dependencies import installation_files
from src.clariden import (ROOT, RuntimeConfig, Supervisor, child_environment, gpu_assignments,
                          load_config, main, memory_estimate, needed_servers, pipeline_command,
                          server_command, server_ready, snapshot_metadata, validate_pipeline)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "containers/deployment.toml", {})

    def test_allocation_mapping_preserves_slurm_order_and_uuid_tokens(self):
        self.assertEqual(gpu_assignments(self.config, "GPU-a,GPU-d,GPU-b,GPU-c"),
                         {"apertus": "GPU-a,GPU-d", "qwen": "GPU-b", "pipeline": "GPU-c"})
        self.assertEqual(gpu_assignments(self.config, "4,6,5,7")["pipeline"], "7")
        self.assertEqual(gpu_assignments(self.config, None)["apertus"], "0,1")
        for visible in ("", "-1", "0,1,2", "0,1,1,3"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                gpu_assignments(self.config, visible)

    def test_env_overrides_and_invalid_allocations(self):
        config = load_config(ROOT / "containers/deployment.toml", {"APERTUS_MAX_MODEL_LEN": "4096",
                       "QWEN_MEMORY_UTILIZATION": "0.85", "APERTUS_EXTRA_ARGS": '["--enforce-eager"]'})
        self.assertEqual(config.apertus.max_model_len, 4096)
        self.assertEqual(config.apertus.extra_args, ["--enforce-eager"])
        self.assertEqual(config.qwen.memory_utilization, .85)
        for environment in ({"QWEN_GPUS": "1"}, {"APERTUS_TENSOR_PARALLEL": "4"},
                            {"QWEN_PORT": "8000"}, {"APERTUS_MEMORY_UTILIZATION": "nan"},
                            {"RUNTIME_STARTUP_TIMEOUT": "0"}, {"QWEN_EXTRA_ARGS": '["--host=0.0.0.0"]'}):
            with self.subTest(env=environment), self.assertRaises(ValueError):
                load_config(ROOT / "containers/deployment.toml", environment)

    def test_child_environment_forces_offline_and_removes_torch_rank(self):
        original = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "RANK": "2", "WORLD_SIZE": "4",
                    "HF_HUB_OFFLINE": "0", "VLLM_CACHE_ROOT": "/cache/vllm"}
        child = child_environment(original, "3", "pipeline")
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(child["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("RANK", child)
        self.assertNotIn("WORLD_SIZE", child)
        self.assertEqual(child["VLLM_CACHE_ROOT"], "/cache/vllm/pipeline")
        self.assertEqual(original["HF_HUB_OFFLINE"], "0")

    def test_commands_bind_localhost_and_pin_perception_to_remapped_gpu(self):
        server = server_command(self.config.apertus, "/cache/snapshot")
        self.assertEqual(server[server.index("--host") + 1], "127.0.0.1")
        self.assertEqual(server[server.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(server[server.index("--model") + 1], "/cache/snapshot")
        self.assertEqual(server[server.index("--compilation-config.pass_config.fuse_allreduce_rms") + 1], "false")
        command = pipeline_command(self.config, ["--scene", "scene_01", "--limit", "1"])
        for flag in ("--dino-device", "--sam-device"):
            self.assertEqual(command[command.index(flag) + 1], "cuda:0")
        for extra in (["--sam-device=cpu"], ["--model", "different-model"]):
            with self.assertRaises(ValueError):
                pipeline_command(self.config, extra)

    def test_replay_only_launches_required_servers(self):
        self.assertEqual(needed_servers(SimpleNamespace(input=None, build_only=True)), ["apertus", "qwen"])
        self.assertEqual(needed_servers(SimpleNamespace(input=Path("saved.json"), build_only=False)), ["apertus"])
        self.assertEqual(needed_servers(SimpleNamespace(input=Path("saved.json"), build_only=True)), [])
        self.assertEqual(needed_servers(SimpleNamespace(input=Path("saved.json"), build_only=True), smoke=True), ["apertus", "qwen"])

    def test_pipeline_validation_fails_before_model_start(self):
        with TemporaryDirectory() as folder:
            data = Path(folder)
            (data / "scene_x").mkdir()
            (data / "scene_x/rgb.png").write_bytes(b"fixture")
            valid = ["--data", folder, "--scene", "scene_x", "--limit", "1"]
            self.assertEqual(validate_pipeline(pipeline_command(self.config, valid)).scene, ["scene_x"])
            for extra in (["--scene", "missing"], ["--limit", "0"], ["--geometry-input", str(data / "absent.json")]):
                with self.assertRaises(ValueError):
                    validate_pipeline(pipeline_command(self.config, ["--data", folder, *extra]))

    def test_dry_run_has_no_torch_imports_or_model_downloads(self):
        with patch("src.clariden.subprocess_json") as probe, patch("src.clariden.resolve_snapshot") as snapshot, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--dry-run", "--", "--limit", "1"]), 0)
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["offline"])
        probe.assert_not_called()
        snapshot.assert_not_called()

    def test_installer_preserves_custom_fork_and_rejects_conflicts(self):
        project = {"project": {"dependencies": ["numpy>=2,<3"],
                             "optional-dependencies": {"vision": ["transformers>=5.10.4,<6"]}}}
        requirements, constraints = installation_files(project, {"transformers": "5.14.0.dev0", "torch": "2.11.0+cu130"})
        self.assertIn("transformers==5.14.0.dev0", constraints)
        self.assertIn("torch==2.11.0+cu130", constraints)
        self.assertNotIn("torch", requirements)
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            installation_files(project, {"transformers": "4.57.0"})


class CacheTests(unittest.TestCase):
    def test_incomplete_cached_shards_are_rejected_without_loading_weights(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"architectures":["Example"]}')
            (path / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 1024}, "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
            (path / "part-1.safetensors").write_bytes(b"test")
            with self.assertRaisesRegex(RuntimeError, "part-2"):
                snapshot_metadata(path)
            (path / "part-2.safetensors").write_bytes(b"test")
            result = snapshot_metadata(path)
            self.assertEqual(result["weight_bytes"], 1024)
            self.assertEqual(result["shards"], 2)

    def test_bf16_weight_and_kv_lower_bound_uses_context_and_tp(self):
        config = load_config(ROOT / "containers/deployment.toml", {})
        model = {"weight_bytes": 140 * 2**30, "config": {"text_config": {
            "num_hidden_layers": 80, "num_key_value_heads": 8, "head_dim": 128}}}
        estimate = memory_estimate(model, config.apertus, [96 * 2**30] * 2)
        self.assertEqual(estimate["weights_per_rank_GiB"], 70)
        self.assertEqual(estimate["kv_per_rank_GiB"], 1.25)
        with self.assertRaisesRegex(RuntimeError, "lower bound"):
            memory_estimate(model, config.apertus, [72 * 2**30] * 2)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.logs = Path(self.directory.name)
        self.supervisor = Supervisor(self.logs, RuntimeConfig(startup_timeout=.1,
                                      shutdown_timeout=.15, poll_interval=.01, monitor_interval=60))

    def tearDown(self):
        self.supervisor.close()
        self.directory.cleanup()

    def start(self, code):
        return self.supervisor.start("fixture", [sys.executable, "-u", "-c", code], dict(os.environ))

    def test_readiness_rejects_wrong_served_model_and_http_failure(self):
        response = io.BytesIO(json.dumps({"data": [{"id": "wrong"}]}).encode())
        response.status = 200
        health = io.BytesIO(b"")
        health.status = 200
        opener = SimpleNamespace(open=unittest.mock.Mock(side_effect=[health, response]))
        with patch("src.clariden.urllib.request.build_opener", return_value=opener):
            self.assertFalse(server_ready(8000, "expected"))
        opener.open.side_effect = OSError("connection refused")
        with patch("src.clariden.urllib.request.build_opener", return_value=opener):
            self.assertFalse(server_ready(8000, "expected"))

    def test_real_local_http_inventory_and_health_then_shutdown(self):
        code = """
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"fixture-model"}]}')
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""
        child = self.start(code)
        deadline = time.monotonic() + 5
        while not child.log_path.read_text().strip() and time.monotonic() < deadline:
            self.supervisor.check_alive([child])
            time.sleep(.01)
        port = int(child.log_path.read_text().strip())
        self.assertTrue(server_ready(port, "fixture-model"))
        self.assertFalse(server_ready(port, "another-model"))
        self.supervisor.close()
        self.assertFalse(server_ready(port, "fixture-model"))

    def test_startup_timeout_stops_before_pipeline(self):
        child = self.start("import time; time.sleep(60)")
        config = load_config(ROOT / "containers/deployment.toml", {})
        with patch("src.clariden.server_ready", return_value=False), patch.object(self.supervisor, "monitor"), \
                self.assertRaises(TimeoutError):
            self.supervisor.wait_ready([(child, config.apertus)])
        self.assertEqual(len(self.supervisor.children), 1)

    def test_failed_child_diagnostic_includes_exit_code_and_log_tail(self):
        child = self.start("print('meaningful engine error'); raise SystemExit(7)")
        child.process.wait(timeout=5)
        with self.assertRaisesRegex(RuntimeError, "meaningful engine error"):
            self.supervisor.check_alive([child])

    def test_server_crash_aborts_pipeline_and_cleanup_reaps_it(self):
        server = self.start("raise SystemExit(9)")
        server.process.wait(timeout=5)
        pipeline = self.supervisor.start("pipeline", [sys.executable, "-c", "import time;time.sleep(60)"], dict(os.environ))
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            self.supervisor.wait_pipeline(pipeline, [server])
        self.supervisor.close()
        self.assertIsNotNone(pipeline.process.poll())
        self.assertTrue(pipeline.stream.closed)

    def test_shutdown_escalates_for_process_ignoring_sigterm(self):
        child = self.start("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(60)")
        deadline = time.monotonic() + 5
        while "ready" not in child.log_path.read_text() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIn("ready", child.log_path.read_text())
        self.supervisor.close()
        self.assertEqual(child.process.returncode, -signal.SIGKILL)

    def test_pipeline_exit_code_is_preserved(self):
        child = self.start("raise SystemExit(23)")
        with patch.object(self.supervisor, "monitor"):
            self.assertEqual(self.supervisor.wait_pipeline(child, []), 23)


if __name__ == "__main__":
    unittest.main()
