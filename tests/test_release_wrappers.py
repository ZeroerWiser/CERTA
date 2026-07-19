import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "01_run_main.sh",
    "02_run_ablation.sh",
    "03_run_efficiency.sh",
    "04_evaluate.sh",
)


class PublicWrapperContractTests(unittest.TestCase):
    def test_each_public_wrapper_has_safe_help(self):
        for name in WRAPPERS:
            wrapper = ROOT / "scripts" / name
            self.assertTrue(wrapper.is_file(), name)
            result = subprocess.run(
                ["bash", str(wrapper), "--help"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")

    def test_main_wrapper_uses_single_output_dir_and_real_limit(self):
        with self._fake_release() as root:
            root = Path(root)
            env = self._run_environment(root)
            env["CERTA_PROFILE"] = "configs/profiles/not-public.env"
            result = subprocess.run(["bash", "scripts/01_run_main.sh"], cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = Path(env["CERTA_OUTPUT_ROOT"]) / "full_cert_run-1"
            captured = json.loads((run_dir / "captured.json").read_text(encoding="utf-8"))
            self.assertEqual(captured["CSCR_OUTPUT_DIR"], str(run_dir))
            self.assertEqual(captured["argv"], ["full_cert", "--limit", "3"])
            self.assertEqual(captured["CSCR_MAIN_CERT_PROFILE"], "0")
            self.assertEqual(captured["CSCR_SEED"], "0")
            self.assertEqual(captured["CSCR_API_CACHE_MODE"], "off")
            self.assertEqual(captured["CSCR_MAX_ANSWER_TOKENS"], "32")
            self.assertEqual(captured["CSCR_MAX_LEN"], "8192")
            self.assertEqual(captured["PYTHONHASHSEED"], "0")
            self.assertTrue((run_dir / "run_config.json").is_file())
            self.assertTrue((run_dir / "release_metadata.json").is_file())
            metadata = json.loads((run_dir / "release_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["main_cert_profile"], "0")
            self.assertEqual(metadata["seed"], "0")
            self.assertEqual(metadata["cache_mode"], "off")
            self.assertEqual(metadata["max_answer_tokens"], "32")
            self.assertEqual(metadata["max_model_len"], "8192")
            self.assertEqual(metadata["pythonhashseed"], "0")

    def test_main_wrapper_rejects_missing_required_runtime_configuration(self):
        with self._fake_release() as root:
            root = Path(root)
            env = self._run_environment(root)
            env.pop("CERTA_API_MODEL")
            result = subprocess.run(["bash", "scripts/01_run_main.sh"], cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("CERTA_API_MODEL", result.stderr)

    def _fake_release(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        shutil.copytree(ROOT / "scripts", root / "scripts")
        (root / "configs/profiles").mkdir(parents=True)
        (root / "configs/profiles/main.env").write_text(
            'export CERTA_SOURCE_COMMIT="0135203cad30710ddd4a854c9228dd564c2fca84"\nexport CERTA_LEGACY_MODE="full_cert"\n',
            encoding="utf-8",
        )
        (root / "input.json").write_text("[]", encoding="utf-8")
        (root / "tables").mkdir()
        (root / "run_cscr.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "FAKE_ARGV=\"$*\" python3 - <<'PY'\nimport json, os\nfrom pathlib import Path\nout = Path(os.environ['CSCR_OUTPUT_DIR'])\nout.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'captured.json').write_text(json.dumps({key: os.environ.get(key) for key in ('CSCR_OUTPUT_DIR', 'CSCR_MAIN_CERT_PROFILE', 'CSCR_SEED', 'CSCR_API_CACHE_MODE', 'CSCR_MAX_ANSWER_TOKENS', 'CSCR_MAX_LEN', 'PYTHONHASHSEED')} | {'argv': os.environ['FAKE_ARGV'].split()}))\n"
            "(out / 'run_config.json').write_text('{}')\n"
            "(out / 'predictions.jsonl').write_text('')\n"
            "(out / 'predictions.debug.jsonl').write_text('')\n"
            "(out / 'metrics.json').write_text('{}')\nPY\n",
            encoding="utf-8",
        )
        os.chmod(root / "run_cscr.sh", 0o755)
        return temp

    def _run_environment(self, root):
        output = root / "runs"
        return {
            **os.environ,
            "CERTA_PYTHON": sys.executable,
            "CERTA_DATASET": "aitqa",
            "CERTA_INPUT_FILE": str(root / "input.json"),
            "CERTA_TABLE_DIR": str(root / "tables"),
            "CERTA_OUTPUT_ROOT": str(output),
            "CERTA_RUN_ID": "run-1",
            "CERTA_GENERATOR_BACKEND": "vllm_chat",
            "CERTA_MODEL_ID": "Qwen3-8B",
            "CERTA_API_BASE_URL": "http://127.0.0.1:30338/v1",
            "CERTA_API_MODEL": "Qwen3-8B",
            "CERTA_LIMIT": "3",
        }


if __name__ == "__main__":
    unittest.main()
