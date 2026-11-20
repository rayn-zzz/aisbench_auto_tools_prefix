"""Regression tests for parsing native metric rows without consuming log messages."""
import logging
from pathlib import Path
import tempfile
import unittest

from save_file import DEFAULT_METRICS, get_data


TABLE = """[ais_bench] Current exp folder: /workspace/example-output
│ Performance Parameters │ Stage │ Average │ Min │ Max │ Median │ P75 │ P90 │ P99 │ N │
│ E2EL │ total │ 6044.8 ms │ 100.0 ms │ 9000.0 ms │ 6050.0 ms │ 7000.0 ms │ 8000.0 ms │ 8500.0 ms │ 15 │
│ TTFT │ total │ 442.8 ms │ 100.0 ms │ 900.0 ms │ 450.0 ms │ 480.0 ms │ 500.6 ms │ 800.0 ms │ 15 │
│ TPOT │ total │ 11.0 ms │ 8.0 ms │ 20.0 ms │ 12.0 ms │ 13.0 ms │ 14.0 ms │ 15.0 ms │ 15 │
│ InputTokens │ total │ 140.0 │ 140.0 │ 140.0 │ 140.0 │ 140.0 │ 140.0 │ 140.0 │ 15 │
│ OutputTokens │ total │ 512.0 │ 512.0 │ 512.0 │ 512.0 │ 512.0 │ 512.0 │ 512.0 │ 15 │
│ Common Metric │ Stage │ Value │
│ Benchmark Duration │ total │ 6044.8031 ms │
│ Total Requests │ total │ 15 │
│ Concurrency │ total │ 3.25 │
│ Max Concurrency │ total │ 5 │
│ Output Token Throughput │ total │ 84.7009 token/s │
│ Input Token Throughput │ total │ 23.1604 token/s │
│ Total Token Throughput │ total │ 107.8612 token/s │
│ Request Throughput │ total │ 2.5 req/s │
│ Prefill Token Throughput │ total │ 100.125 token/s │
"""

EXPECTED = {
    "input_len": 140.0, "output_len": 512.0, "total_req": 15.0,
    "max_cc": "5", "cc": 3.25, "rr": 0.75, "repeat_rate": 0.6,
    "TTFT avg": 442.8, "TTFT P90": 500.6,
    "TPOT avg": 11.0, "TPOT SLO_P90": 14.0,
    "E2E_time": 6.0448031, "output_throughput": 84.7009,
    "single_output_throughput": 84.7009 / 4,
    "E2E_throughput": 107.8612, "single_E2E_throughput": 107.8612 / 4,
    "qps": 2.5, "qpm": 150.0, "input_token_throughput": 23.1604,
    "prefill_token_throughput": 100.125,
    "E2EL avg": 6044.8, "E2EL P90": 8000.0,
}


class GetDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.log_path = Path(self.temporary.name) / "aisbench.log"

    def parse(self, text):
        self.log_path.write_text(text, encoding="utf-8")
        return get_data(str(self.log_path), 0.75, 4, 0.6)

    def assert_metrics(self, metrics):
        self.assertEqual(set(metrics), set(EXPECTED) | {"current_time"})
        self.assertIsInstance(metrics["current_time"], str)
        for field, expected in EXPECTED.items():
            with self.subTest(field=field):
                if isinstance(expected, float):
                    self.assertAlmostEqual(metrics[field], expected)
                else:
                    self.assertEqual(metrics[field], expected)

    def test_unicode_table_extracts_metrics_and_conversions(self):
        metrics, directory = self.parse(TABLE)
        self.assert_metrics(metrics)
        self.assertEqual(directory, "/workspace/example-output")

    def test_ascii_table_has_the_same_metrics(self):
        metrics, _ = self.parse(TABLE.replace("│", "|"))
        self.assert_metrics(metrics)

    def test_ansi_wrapped_rows_have_the_same_metrics(self):
        colored = "\n".join("\x1b[32m" + row + "\x1b[0m" for row in TABLE.splitlines())
        metrics, _ = self.parse(colored)
        self.assert_metrics(metrics)

    def test_repeated_reads_ignore_the_dictionary_logged_to_the_same_file(self):
        self.log_path.write_text(TABLE, encoding="utf-8")
        handler = logging.FileHandler(str(self.log_path), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            for attempt in range(3):
                with self.subTest(attempt=attempt):
                    metrics, _ = get_data(str(self.log_path), 0.75, 4, 0.6)
                    handler.flush()
                    self.assert_metrics(metrics)
            self.assertEqual(self.log_path.read_text(encoding="utf-8").count("'E2EL avg'"), 3)
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_dictionary_without_tables_keeps_default_measurements(self):
        metrics, _ = self.parse("INFO " + repr(dict(EXPECTED)) + "\n")
        for field, default in DEFAULT_METRICS.items():
            if field not in {"current_time", "rr", "repeat_rate"}:
                with self.subTest(field=field):
                    self.assertEqual(metrics[field], default)

    def test_numeric_log_messages_and_unknown_labels_do_not_override_rows(self):
        noise = """WARNING E2EL 1.0 2.0 3.0 4.0 5.0 6.0
INFO TTFT 1.0 2.0 3.0 4.0 5.0 6.0
WARNING Max Concurrency changed to 99
INFO Output Token Throughput is 999.0
│ E2EL details │ total │ 1.0 2.0 3.0 4.0 5.0 6.0 │
│ Max Concurrency hint │ total │ 99 │
"""
        metrics, _ = self.parse(TABLE + noise)
        self.assert_metrics(metrics)

    def test_maximum_concurrency_is_distinct_from_average_concurrency(self):
        metrics, _ = self.parse("│ Max Concurrency │ total │ 5 │\n│ Concurrency │ total │ 3.25 │\n")
        self.assertEqual(metrics["max_cc"], "5")
        self.assertEqual(metrics["cc"], 3.25)

    def test_last_valid_table_wins_over_previous_results_and_dictionaries(self):
        next_table = TABLE.replace("6044.8 ms", "7044.8 ms").replace("8000.0 ms", "8100.0 ms")
        metrics, _ = self.parse(TABLE + "INFO " + repr(EXPECTED) + "\n" + next_table)
        expected = dict(EXPECTED, **{"E2EL avg": 7044.8, "E2EL P90": 8100.0})
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertEqual(metrics[field], value)


if __name__ == "__main__":
    unittest.main()
