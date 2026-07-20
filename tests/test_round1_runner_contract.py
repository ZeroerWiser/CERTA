import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "configs" / "profiles" / "certa_round1_shadow.env"
RUNNER = REPO_ROOT / "scripts" / "05_run_round1_shadow.sh"


class Round1RunnerContractTests(unittest.TestCase):
    def _profile_environment(self):
        completed = subprocess.run(
            ["bash", "-c", 'set -a; source "$1"; env -0', "bash", str(PROFILE)],
            check=True,
            capture_output=True,
        )
        return dict(item.split("=", 1) for item in completed.stdout.decode().split("\0") if item)

    def test_profile_pins_backbone_sampling_and_disables_every_mutator(self):
        env = self._profile_environment()
        expected = {
            "CSCR_DATASET": "hitab",
            "CSCR_GENERATOR_BACKEND": "vllm_chat",
            "CSCR_API_BASE_URL": "http://127.0.0.1:30338/v1",
            "CSCR_API_MODEL": "Qwen3-8B",
            "CSCR_API_KEY_ENV": "EMPTY",
            "CSCR_API_CACHE_MODE": "readwrite",
            "CSCR_MAX_LEN": "32768",
            "CSCR_MAX_ANSWER_TOKENS": "32",
            "CSCR_TEMPERATURE": "0.0",
            "CSCR_TOP_P": "1.0",
            "CSCR_SEED": "0",
            "CSCR_API_MAX_RETRIES": "0",
            "CSCR_SAVE_LLM_INPUTS": "hash",
            "CSCR_MAIN_CERT_PROFILE": "1",
            "CSCR_OPERATION_COMMIT_GATE_MODE": "diagnostic",
            "CSCR_OPERATION_COMMIT_VERSION": "E67",
            "CSCR_BLACK_BOX_COMMIT_POLICY": "certified",
            "CSCR_ADAPTIVE_PROMPT": "0",
            "CSCR_CREDAL_PROBE": "0",
            "CSCR_CREDAL_GATE": "0",
            "CSCR_QUESTION_TYPE_ROUTER": "0",
            "CSCR_ONLINE_NORMALIZER": "0",
            "CSCR_ORACLE_ONLINE_NORMALIZER": "0",
            "CSCR_API_FORMAT_NORMALIZER": "off",
            "CSCR_HCEG_FALLBACK": "0",
            "CSCR_CERT_COMMIT_BOUNDARY": "0",
            "CSCR_SELF_CONSISTENCY": "0",
            "CSCR_SOURCE_RISK_CALIBRATION": "off",
        }
        self.assertEqual({key: env.get(key) for key in expected}, expected)

    def test_runner_is_shadow_only_and_bounds_diagnostic_arms(self):
        text = RUNNER.read_text(encoding="utf-8")
        required = (
            "--enable-cera-repair",
            "--cera-stage E71",
            "--cera-shadow-only",
            "--cera-round6-e71-v4",
            "--cera-enable-typed-planner",
            "--cera-planner-contract rcpc_signature_v2",
            "--cera-planner-legacy-query-semantics-mode audit_only",
            "--cera-log-evidence-packet",
            "proposal_blind_schema_only",
            "proposal_blind_value_aware",
            "proposal_aware_diagnostic",
            "--limit 8",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        for forbidden in ("--cera-stepwise-trace", "--cera-commit-approved-repair", "Judge", "judge"):
            self.assertNotIn(forbidden, text)
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)


if __name__ == "__main__":
    unittest.main()
