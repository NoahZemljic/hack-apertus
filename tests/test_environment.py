"""Environment failures should be explicit without loading models or calling servers."""

from contextlib import redirect_stdout
from importlib import metadata, util
import io
import json
import unittest
from unittest.mock import patch

from src.check_environment import check_environment, main


class EnvironmentTests(unittest.TestCase):
    def test_namespace_package_and_base_operations_work(self) -> None:
        report = check_environment(profile="base")
        self.assertTrue(report["ok"], report)
        self.assertTrue(any(item["check"] == "project import: src.masking.sam" for item in report["checks"]))

    def test_missing_dependency_is_a_named_failure_and_cli_exits_nonzero(self) -> None:
        real_version = metadata.version

        def installed(name: str) -> str:
            if name == "Pillow":
                raise metadata.PackageNotFoundError(name)
            return real_version(name)

        stream = io.StringIO()
        with patch("src.check_environment.metadata.version", side_effect=installed), redirect_stdout(stream):
            status = main(["--profile", "base", "--json"])
        report = json.loads(stream.getvalue())
        self.assertEqual(status, 1)
        failures = [item for item in report["checks"] if not item["ok"]]
        self.assertEqual(len(failures), 1)
        self.assertIn("Pillow", failures[0]["check"])
        self.assertIn("PackageNotFoundError", failures[0]["detail"])

    @unittest.skipUnless(
        all(util.find_spec(name) is not None for name in ("torch", "torchvision", "transformers", "openai")),
        "Requires the vision and accelerator dependencies",
    )
    def test_cuda_requirement_cannot_pass_on_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            report = check_environment(require_cuda=True)
        failures = [item for item in report["checks"] if not item["ok"]]
        self.assertEqual(len(failures), 1, report)
        self.assertEqual(failures[0]["check"], "Torch/Torchvision operators")
        self.assertIn("CUDA is unavailable", failures[0]["detail"])

    def test_server_inventory_checks_name_without_generating(self) -> None:
        request_urls = []

        def respond(request, **kwargs):
            request_urls.append(request.full_url)
            self.assertIsNone(request.data)
            return io.BytesIO(json.dumps({"data": [{"id": "served-model"}]}).encode())

        with patch("urllib.request.urlopen", side_effect=respond):
            report = check_environment(profile="base", servers=[("test", "http://localhost:1234/v1", "different-model")])
        self.assertFalse(report["ok"])
        self.assertEqual(request_urls, ["http://localhost:1234/v1/models"])
        self.assertIn("available", report["checks"][-1]["detail"])

    def test_unsupported_python_stops_before_dependency_probes(self) -> None:
        with patch("src.check_environment.sys.version_info", (3, 10)), patch("src.check_environment.metadata.version") as version:
            report = check_environment(profile="base")
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["checks"]), 1)
        version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
