import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ostinato.viewpoints import Observation, demo, load_pair, summarize


class ViewpointTests(unittest.TestCase):
    def test_reference_accounts_for_elapsed_time(self):
        row = Observation("front", 40, 17, 0.9, "ok", 10)
        self.assertEqual(row.reference_bpm, 15)

    def test_recovery_and_coverage_keep_declined_windows(self):
        first = Observation("near", 30, None, 0, "obstructed", 8)
        second = Observation("far", 30, 18, 0.9, "ok", 8)
        report = summarize([(first, second)])
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["mean_absolute_error_bpm_on_reported"], 2)
        self.assertEqual(report["second_view_reports_after_first_decline"], 1)

    def test_no_accepted_reading_has_no_error_estimate(self):
        row = Observation("A", 30, None, 0, "weak", 7)
        self.assertIsNone(summarize([(row, row)])["mean_absolute_error_bpm_on_reported"])

    def test_simulation_includes_travel_observation_and_failure(self):
        runs = demo()["runs"]
        times = [run["time_to_reading_s"] for run in runs]
        self.assertEqual(times, [64, 34, 31, 31, 31, 67, None, None])
        self.assertEqual([run["spent_s"] for run in runs[-2:]], [64, 67])

    def test_simulated_results_rejected_by_webcam_report(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            path.write_text(json.dumps(demo()))
            with self.assertRaisesRegex(ValueError, "paired webcam"):
                load_pair(path)
