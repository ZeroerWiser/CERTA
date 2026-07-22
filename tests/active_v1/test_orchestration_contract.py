import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class ActiveOrchestrationContractTests(unittest.TestCase):
    def test_tool_direct_entrypoint_resolves_repository_imports(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "tools/certa_active_v1.py"), "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capability-fixtures", result.stdout)


if __name__ == "__main__":
    unittest.main()
