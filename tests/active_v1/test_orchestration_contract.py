import subprocess
import sys
import unittest
from pathlib import Path

from tools import certa_active_v1


REPO = Path(__file__).resolve().parents[2]


class ActiveOrchestrationContractTests(unittest.TestCase):
    def test_role_interface_schema_version_matches_immutable_pack(self):
        pack_schema = certa_active_v1.PACK / "schemas/INTERFACE_FREEZE_SCHEMA.json"
        expected = __import__("json").loads(pack_schema.read_text(encoding="utf-8"))["properties"]["schema_version"]["const"]
        self.assertEqual(getattr(certa_active_v1, "ROLE_INTERFACE_SCHEMA_VERSION", None), expected)

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
