"""Tests for the new repository classes and TrainingService.

All tests use mock psycopg2 connections so no live database is required.
"""

import unittest
from unittest import mock

from app.models.correction import Correction
from app.models.title_back_record import TitleBackRecord
from app.models.training_run import TrainingRun
from app.repositories.correction_repository import CorrectionRepository
from app.repositories.title_back_record_repository import TitleBackRecordRepository
from app.repositories.training_run_repository import TrainingRunRepository
from app.services.training_service import TrainingService

_GOOD_VIN = "1HGCM82633A004352"


# ---------------------------------------------------------------------------
# Shared mock-connection builder
# ---------------------------------------------------------------------------


def _make_connection(cursor_rows=None, fetchone_return=None, fetchone_explicit_none=False):
    """Return a mock psycopg2-style connection/cursor pair."""
    cursor = mock.MagicMock()
    if fetchone_explicit_none:
        cursor.fetchone.return_value = None
    else:
        cursor.fetchone.return_value = fetchone_return or (1,)
    cursor.fetchall.return_value = cursor_rows or []

    cm = mock.MagicMock()
    cm.__enter__ = mock.Mock(return_value=cursor)
    cm.__exit__ = mock.Mock(return_value=False)

    connection = mock.MagicMock()
    connection.cursor.return_value = cm
    return connection, cursor


# ---------------------------------------------------------------------------
# CorrectionRepository tests
# ---------------------------------------------------------------------------


class CorrectionRepositoryCreateTests(unittest.TestCase):
    def test_create_inserts_and_returns_correction(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(5,))
        repo = CorrectionRepository(connection)
        result = repo.create(
            record_id=42,
            field_name="vin",
            original_value="BAD",
            corrected_value=_GOOD_VIN,
            is_ground_truth=True,
        )
        self.assertIsInstance(result, Correction)
        self.assertEqual(5, result.id)
        self.assertEqual(42, result.record_id)
        self.assertEqual("vin", result.field_name)
        self.assertEqual(_GOOD_VIN, result.corrected_value)
        self.assertTrue(result.is_ground_truth)
        connection.commit.assert_called_once()

    def test_create_passes_ground_truth_flag_as_integer(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(1,))
        CorrectionRepository(connection).create(
            record_id=1, field_name="make",
            original_value=None, corrected_value="FORD", is_ground_truth=True,
        )
        params = cursor.execute.call_args[0][1]
        # is_ground_truth is 6th positional param (0-indexed = 5)
        self.assertEqual(1, params[5])

    def test_create_with_image_hash(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(3,))
        result = CorrectionRepository(connection).create(
            record_id=10, field_name="color",
            original_value="BLUE", corrected_value="RED",
            image_hash="/app/data/img.png",
        )
        self.assertEqual("/app/data/img.png", result.image_hash)

    def test_create_sql_inserts_into_corrections(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(9,))
        CorrectionRepository(connection).create(
            record_id=7, field_name="model",
            original_value=None, corrected_value="ACCORD",
        )
        sql = cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO corrections", sql)
        self.assertIn("RETURNING id", sql)


class CorrectionRepositoryApproveTests(unittest.TestCase):
    def test_approve_executes_update(self) -> None:
        connection, cursor = _make_connection()
        CorrectionRepository(connection).approve(17)
        sql = cursor.execute.call_args[0][0]
        self.assertIn("UPDATE corrections", sql)
        self.assertIn("is_ground_truth = 1", sql)
        params = cursor.execute.call_args[0][1]
        self.assertEqual(17, params[0])
        connection.commit.assert_called_once()


class CorrectionRepositoryListTests(unittest.TestCase):
    def test_list_for_record_returns_corrections(self) -> None:
        fake_rows = [
            {
                "id": 1, "record_id": 42, "image_hash": None,
                "field_name": "vin", "original_value": "BAD",
                "corrected_value": _GOOD_VIN, "is_ground_truth": 1,
                "created_at": "2024-01-01",
            }
        ]
        connection, _ = _make_connection(cursor_rows=fake_rows)
        corrections = CorrectionRepository(connection).list_for_record(42)
        self.assertEqual(1, len(corrections))
        self.assertIsInstance(corrections[0], Correction)
        self.assertEqual("vin", corrections[0].field_name)
        self.assertTrue(corrections[0].is_ground_truth)

    def test_list_for_record_returns_empty_when_none(self) -> None:
        connection, _ = _make_connection(cursor_rows=[])
        self.assertEqual([], CorrectionRepository(connection).list_for_record(99))

    def test_list_ground_truth_joins_source_file_path(self) -> None:
        fake_rows = [
            {
                "id": 2, "record_id": 5, "image_hash": None,
                "field_name": "make", "original_value": None,
                "corrected_value": "FORD", "is_ground_truth": 1,
                "created_at": "2024-02-01",
                "source_file_path": "/app/data/uploads/abc.png",
            }
        ]
        connection, cursor = _make_connection(cursor_rows=fake_rows)
        corrections = CorrectionRepository(connection).list_ground_truth()
        self.assertEqual(1, len(corrections))
        self.assertEqual("/app/data/uploads/abc.png", corrections[0].source_file_path)
        # Verify the JOIN was part of the query
        sql = cursor.execute.call_args[0][0]
        self.assertIn("source_file_path", sql)
        self.assertIn("is_ground_truth = 1", sql)


# ---------------------------------------------------------------------------
# TitleBackRecordRepository tests
# ---------------------------------------------------------------------------


class TitleBackRecordRepositoryTests(unittest.TestCase):
    def test_save_inserts_and_returns_back_record(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(7,))
        repo = TitleBackRecordRepository(connection)
        result = repo.save(
            TitleBackRecord(
                title_record_id=42,
                odometer_at_sale=95000,
                buyer_name="BUYER CORP",
                sale_price=5000.0,
                sale_date="2024-06-01",
            )
        )
        self.assertIsInstance(result, TitleBackRecord)
        self.assertEqual(7, result.id)
        self.assertEqual(42, result.title_record_id)
        self.assertEqual("BUYER CORP", result.buyer_name)
        connection.commit.assert_called_once()

    def test_save_sql_contains_upsert(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(1,))
        TitleBackRecordRepository(connection).save(TitleBackRecord(title_record_id=1))
        sql = cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO title_back_records", sql)
        self.assertIn("ON CONFLICT", sql)
        self.assertIn("RETURNING id", sql)

    def test_save_sets_created_at_when_missing(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(3,))
        result = TitleBackRecordRepository(connection).save(TitleBackRecord(title_record_id=5))
        self.assertNotEqual("", result.created_at)

    def test_save_preserves_existing_created_at(self) -> None:
        connection, _ = _make_connection(fetchone_return=(4,))
        record = TitleBackRecord(title_record_id=6, created_at="2020-01-01T00:00:00")
        result = TitleBackRecordRepository(connection).save(record)
        self.assertEqual("2020-01-01T00:00:00", result.created_at)

    def test_get_by_title_record_id_returns_model(self) -> None:
        fake_row = {
            "id": 9, "title_record_id": 42, "odometer_at_sale": 85000,
            "buyer_name": "BUYER LLC", "buyer_address": None,
            "seller_name": None, "sale_price": None,
            "sale_date": "2024-05-01", "notes": None,
            "created_at": "2024-05-01T12:00:00",
        }
        connection, _ = _make_connection(fetchone_return=fake_row)
        result = TitleBackRecordRepository(connection).get_by_title_record_id(42)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, TitleBackRecord)
        self.assertEqual(9, result.id)
        self.assertEqual("BUYER LLC", result.buyer_name)

    def test_get_by_title_record_id_returns_none_when_missing(self) -> None:
        connection, _ = _make_connection(fetchone_explicit_none=True)
        result = TitleBackRecordRepository(connection).get_by_title_record_id(999)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TrainingRunRepository tests
# ---------------------------------------------------------------------------


class TrainingRunRepositoryTests(unittest.TestCase):
    def test_create_returns_run_with_id(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(11,))
        repo = TrainingRunRepository(connection)
        result = repo.create(TrainingRun(sample_count=30, notes="batch A"))
        self.assertIsInstance(result, TrainingRun)
        self.assertEqual(11, result.id)
        self.assertEqual(30, result.sample_count)
        self.assertEqual("batch A", result.notes)
        connection.commit.assert_called_once()

    def test_create_sql_inserts_into_training_runs(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(1,))
        TrainingRunRepository(connection).create(TrainingRun(sample_count=5))
        sql = cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO training_runs", sql)
        self.assertIn("RETURNING id", sql)

    def test_create_sets_exported_at_when_empty(self) -> None:
        connection, _ = _make_connection(fetchone_return=(2,))
        result = TrainingRunRepository(connection).create(TrainingRun(sample_count=1))
        self.assertNotEqual("", result.exported_at)

    def test_list_returns_training_runs(self) -> None:
        fake_rows = [
            {"id": 3, "exported_at": "2024-01-10", "sample_count": 15,
             "notes": "run 1", "export_path": None},
            {"id": 2, "exported_at": "2024-01-05", "sample_count": 8,
             "notes": None, "export_path": "/exports/run2.zip"},
        ]
        connection, cursor = _make_connection(cursor_rows=fake_rows)
        runs = TrainingRunRepository(connection).list()
        self.assertEqual(2, len(runs))
        self.assertIsInstance(runs[0], TrainingRun)
        self.assertEqual(3, runs[0].id)
        self.assertEqual(15, runs[0].sample_count)
        # Verify ORDER BY DESC is in the query
        sql = cursor.execute.call_args[0][0]
        self.assertIn("ORDER BY id DESC", sql)

    def test_list_returns_empty_when_no_runs(self) -> None:
        connection, _ = _make_connection(cursor_rows=[])
        self.assertEqual([], TrainingRunRepository(connection).list())


# ---------------------------------------------------------------------------
# TrainingService tests
# ---------------------------------------------------------------------------


class TrainingServiceAnnotationQueueTests(unittest.TestCase):
    def test_get_annotation_queue_returns_rows(self) -> None:
        fake_rows = [
            {
                "id": 1, "state": "NM", "title_number": "T001",
                "vin": _GOOD_VIN, "vehicle_year": 2003,
                "make": None, "model": None, "color": None,
                "is_validated": 0, "created_at": "2024-01-01", "gt_count": 0,
            }
        ]
        connection, _ = _make_connection(cursor_rows=fake_rows)
        result = TrainingService(connection).get_annotation_queue(limit=10, offset=0)
        self.assertEqual(1, len(result))
        self.assertEqual(0, result[0]["gt_count"])
        self.assertEqual("NM", result[0]["state"])

    def test_get_annotation_queue_returns_empty_list(self) -> None:
        connection, _ = _make_connection(cursor_rows=[])
        self.assertEqual([], TrainingService(connection).get_annotation_queue())

    def test_get_annotation_queue_passes_limit_offset(self) -> None:
        connection, cursor = _make_connection(cursor_rows=[])
        TrainingService(connection).get_annotation_queue(limit=25, offset=50)
        params = cursor.execute.call_args[0][1]
        self.assertEqual(25, params[0])
        self.assertEqual(50, params[1])


class TrainingServiceStatsTests(unittest.TestCase):
    def _make_stats_connection(self):
        """Build a mock that cycles through four cursor calls."""
        cursor = mock.MagicMock()
        cursor.fetchone.side_effect = [
            {"total": 12},
            {"total_records": 8, "ge1": 5, "ge5": 2, "ge10": 0},
        ]
        cursor.fetchall.side_effect = [
            [{"field_name": "vin", "count": 6}, {"field_name": "make", "count": 4}],
            [{"state": "NM", "count": 9}, {"state": "TX", "count": 3}],
        ]
        cm = mock.MagicMock()
        cm.__enter__ = mock.Mock(return_value=cursor)
        cm.__exit__ = mock.Mock(return_value=False)
        connection = mock.MagicMock()
        connection.cursor.return_value = cm
        return connection

    def test_get_stats_returns_all_keys(self) -> None:
        connection = self._make_stats_connection()
        stats = TrainingService(connection).get_stats()
        self.assertIn("total_gt_corrections", stats)
        self.assertIn("by_field", stats)
        self.assertIn("by_state", stats)
        self.assertIn("coverage", stats)

    def test_get_stats_total(self) -> None:
        stats = TrainingService(self._make_stats_connection()).get_stats()
        self.assertEqual(12, stats["total_gt_corrections"])

    def test_get_stats_by_field_length(self) -> None:
        stats = TrainingService(self._make_stats_connection()).get_stats()
        self.assertEqual(2, len(stats["by_field"]))
        self.assertEqual("vin", stats["by_field"][0]["field_name"])

    def test_get_stats_by_state_length(self) -> None:
        stats = TrainingService(self._make_stats_connection()).get_stats()
        self.assertEqual(2, len(stats["by_state"]))

    def test_get_stats_coverage(self) -> None:
        stats = TrainingService(self._make_stats_connection()).get_stats()
        coverage = stats["coverage"]
        self.assertEqual(8, coverage["total_records"])
        self.assertEqual(5, coverage["ge1"])
        self.assertEqual(2, coverage["ge5"])
        self.assertEqual(0, coverage["ge10"])


if __name__ == "__main__":
    unittest.main()
