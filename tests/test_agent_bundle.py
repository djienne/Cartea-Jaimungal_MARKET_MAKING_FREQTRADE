"""One end-to-end check: python -m unittest discover -s tests -p test_agent_bundle.py."""

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_agent_bundle import REFERENCE_BOOK, create_bundle


class AgentBundleTest(unittest.TestCase):
    def test_current_source_references_and_paper_history_without_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            files = {
                "tracked.rs": "old",
                ".env": "SECRET_DO_NOT_EXPORT",
                ".gitignore": "rust_live/reports/\n*.pdf\n",
                REFERENCE_BOOK: "reference book bytes",
                "docs/market_making_introduction.ipynb": '{"nbformat":4}',
                "rust_live/reports/grid_live/grid_state.json": '{"run_id":"run-123"}',
                "rust_live/reports/grid_live/leaderboard.json": '{"rows":[]}',
                "rust_live/reports/grid_live/runs/run-123/equity_history.csv": "time,pnl\n1,2\npartial",
                "rust_live/reports/private_account.json": "PRIVATE_DO_NOT_EXPORT",
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data, encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.rs", ".env"], check=True)
            (root / "tracked.rs").write_text("edited", encoding="utf-8")
            (root / "new_test.py").write_text("new", encoding="utf-8")
            output = root / "review.zip"
            create_bundle(root, output)
            with zipfile.ZipFile(output) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertEqual(bundle.read("project/tracked.rs"), b"edited")
                self.assertEqual(bundle.read("project/new_test.py"), b"new")
                self.assertIn("project/" + REFERENCE_BOOK, bundle.namelist())
                self.assertIn("project/docs/market_making_introduction.ipynb", bundle.namelist())
                self.assertNotIn("project/.env", bundle.namelist())
                self.assertNotIn("project/review.zip", bundle.namelist())
                self.assertNotIn("project/rust_live/reports/private_account.json", bundle.namelist())
                self.assertEqual(bundle.read("project/rust_live/reports/grid_live/runs/run-123/equity_history.csv"), b"time,pnl\n1,2\n")
            self.assertEqual((root / "rust_live/reports/grid_live/runs/run-123/equity_history.csv").read_text(), "time,pnl\n1,2\npartial")


if __name__ == "__main__":
    unittest.main()
