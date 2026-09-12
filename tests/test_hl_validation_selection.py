"""Run with python -m unittest discover -s tests -p test_hl_validation_selection.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_hl_data import iter_parquet_files, validate_symbol


class ShardSelectionTests(unittest.TestCase):
    def test_timestamped_shards_need_no_stat_and_legacy_names_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = root / "prices"
            stream.mkdir()
            for name in ("prices_1000.parquet", "prices_compact_3000.parquet", "legacy.parquet"):
                (stream / name).touch()
            os.utime(stream / "prices_1000.parquet", (9, 9))
            os.utime(stream / "legacy.parquet", (2, 2))
            real_stat = Path.stat

            def stat(path, *args, **kwargs):
                if path.name.startswith("prices_"):
                    raise AssertionError("timestamped shard selection must not stat files")
                return real_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", stat):
                chosen = list(iter_parquet_files(root, ["prices"], newest_per_stream=2))
            self.assertEqual([p.name for _, p in chosen], ["prices_compact_3000.parquet", "legacy.parquet"])

    def test_fresh_names_do_not_hide_stale_contents_or_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stream in ("prices", "trades", "orderbooks"):
                folder = root / "CASHCAT" / stream
                folder.mkdir(parents=True)
                pd.DataFrame({"timestamp": [1.0], "price": [0.18]}).to_parquet(
                    folder / f"{stream}_9999999999999.parquet", index=False
                )
            result = validate_symbol(root, "CASHCAT", newest_per_stream=1, max_age_seconds=180)
            self.assertEqual(result["checked_files"], 3)
            self.assertFalse(result["ok"])
            self.assertFalse(result["fresh"])
            (root / "CASHCAT" / "trades" / "trades_9999999999999.parquet").write_text("broken")
            result = validate_symbol(root, "CASHCAT", newest_per_stream=1)
            self.assertFalse(result["ok"])
            self.assertEqual(result["bad_file_count"], 1)
            self.assertFalse(validate_symbol(root, "MISSING")["ok"])


if __name__ == "__main__":
    unittest.main()
