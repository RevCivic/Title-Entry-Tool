"""Tests for the app.models package (Phase 1 – data models)."""

import unittest
from datetime import datetime

from app.models.correction import Correction
from app.models.title_back_record import TitleBackRecord
from app.models.title_image import TitleImage
from app.models.title_record import (
    ALL_FIELDS,
    CORE_FIELDS,
    EXTENDED_FIELDS,
    FIELD_LABELS,
    OPERATIONAL_FIELDS,
    TitleRecord,
    validate_record,
    validate_title_number,
    validate_vehicle_year,
    validate_vin,
)
from app.models.training_run import TrainingRun

_GOOD_VIN = "1HGCM82633A004352"


class FieldConstantsTests(unittest.TestCase):
    """CORE_FIELDS, EXTENDED_FIELDS, ALL_FIELDS, OPERATIONAL_FIELDS, FIELD_LABELS."""

    def test_core_fields_contains_four_anchors(self) -> None:
        self.assertEqual(
            ("state", "title_number", "vin", "vehicle_year"), CORE_FIELDS
        )

    def test_extended_fields_contains_ten_ai_fields(self) -> None:
        self.assertEqual(10, len(EXTENDED_FIELDS))
        self.assertIn("make", EXTENDED_FIELDS)
        self.assertIn("issue_date", EXTENDED_FIELDS)

    def test_all_fields_is_core_plus_extended(self) -> None:
        self.assertEqual(CORE_FIELDS + EXTENDED_FIELDS, ALL_FIELDS)
        self.assertEqual(14, len(ALL_FIELDS))

    def test_operational_fields_has_ten_entries(self) -> None:
        self.assertEqual(10, len(OPERATIONAL_FIELDS))
        self.assertIn("provider_id", OPERATIONAL_FIELDS)
        self.assertIn("sold_to", OPERATIONAL_FIELDS)

    def test_operational_fields_disjoint_from_all_fields(self) -> None:
        self.assertEqual(set(), set(ALL_FIELDS) & set(OPERATIONAL_FIELDS))

    def test_field_labels_covers_all_nmvitis_fields(self) -> None:
        self.assertEqual(set(ALL_FIELDS), set(FIELD_LABELS.keys()))

    def test_no_duplicates_in_all_fields(self) -> None:
        self.assertEqual(len(ALL_FIELDS), len(set(ALL_FIELDS)))

    def test_no_duplicates_in_operational_fields(self) -> None:
        self.assertEqual(len(OPERATIONAL_FIELDS), len(set(OPERATIONAL_FIELDS)))


class ValidateTitleNumberTests(unittest.TestCase):
    def test_valid_alphanumeric_title_number(self) -> None:
        self.assertIsNone(validate_title_number("ABC1234"))

    def test_valid_with_hyphens_stripped(self) -> None:
        # Hyphens are stripped; remaining 7 chars are valid.
        self.assertIsNone(validate_title_number("ABC-1234"))

    def test_too_short_returns_error(self) -> None:
        error = validate_title_number("AB")
        self.assertIsNotNone(error)
        self.assertIn("3 alphanumeric", error)

    def test_too_long_returns_error(self) -> None:
        error = validate_title_number("A" * 21)
        self.assertIsNotNone(error)
        self.assertIn("20 alphanumeric", error)

    def test_exactly_three_chars_is_valid(self) -> None:
        self.assertIsNone(validate_title_number("ABC"))

    def test_exactly_twenty_chars_is_valid(self) -> None:
        self.assertIsNone(validate_title_number("A" * 20))


class ValidateVinTests(unittest.TestCase):
    def test_valid_vin_passes(self) -> None:
        self.assertIsNone(validate_vin(_GOOD_VIN))

    def test_short_vin_returns_error(self) -> None:
        error = validate_vin("1HGCM82633A00435")
        self.assertIsNotNone(error)
        self.assertIn("17 characters", error)

    def test_vin_with_invalid_character_returns_error(self) -> None:
        vin_with_i = "1HGCM82633A00435I"  # 'I' is not valid
        error = validate_vin(vin_with_i)
        self.assertIsNotNone(error)
        self.assertIn("invalid characters", error)

    def test_vin_with_bad_check_digit_returns_error(self) -> None:
        bad_check = _GOOD_VIN[:8] + "0" + _GOOD_VIN[9:]  # replace pos 8 check digit
        if bad_check == _GOOD_VIN:
            bad_check = _GOOD_VIN[:8] + "1" + _GOOD_VIN[9:]
        # Only run assertion if we actually produced a different VIN.
        if bad_check != _GOOD_VIN:
            error = validate_vin(bad_check)
            self.assertIsNotNone(error)
            self.assertIn("check digit", error)

    def test_vin_with_lowercase_is_accepted_after_normalisation(self) -> None:
        self.assertIsNone(validate_vin(_GOOD_VIN.lower()))


class ValidateVehicleYearTests(unittest.TestCase):
    def test_valid_year_passes(self) -> None:
        self.assertIsNone(validate_vehicle_year(2003))

    def test_year_before_1886_returns_error(self) -> None:
        error = validate_vehicle_year(1885)
        self.assertIsNotNone(error)
        self.assertIn("1886", error)

    def test_year_1886_passes(self) -> None:
        self.assertIsNone(validate_vehicle_year(1886))

    def test_future_year_one_ahead_passes(self) -> None:
        next_year = datetime.utcnow().year + 1
        self.assertIsNone(validate_vehicle_year(next_year))

    def test_future_year_two_ahead_returns_error(self) -> None:
        too_far = datetime.utcnow().year + 2
        error = validate_vehicle_year(too_far)
        self.assertIsNotNone(error)
        self.assertIn("future", error)


class ValidateRecordTests(unittest.TestCase):
    def test_all_valid_inputs_return_no_errors(self) -> None:
        errors = validate_record("ABC1234", _GOOD_VIN, 2003)
        self.assertEqual([], errors)

    def test_three_bad_inputs_return_three_errors(self) -> None:
        errors = validate_record("12", "1HGCM82633A00435I", 1700)
        self.assertEqual(3, len(errors))

    def test_bad_vin_returns_one_error(self) -> None:
        errors = validate_record("ABC1234", "BADVIN", 2003)
        self.assertEqual(1, len(errors))
        self.assertIn("17 characters", errors[0])


class TitleRecordDataclassTests(unittest.TestCase):
    def test_default_instance_has_empty_strings_and_none(self) -> None:
        record = TitleRecord()
        self.assertEqual("", record.state)
        self.assertEqual("", record.vin)
        self.assertEqual(0, record.vehicle_year)
        self.assertIsNone(record.make)
        self.assertFalse(record.is_validated)
        self.assertEqual([], record.validation_errors)
        self.assertIsNone(record.id)

    def test_validate_method_delegates_to_module_function(self) -> None:
        record = TitleRecord(
            state="NM",
            title_number="ABC1234",
            vin=_GOOD_VIN,
            vehicle_year=2003,
        )
        self.assertEqual([], record.validate())

    def test_validate_method_returns_errors_for_invalid_fields(self) -> None:
        record = TitleRecord(state="NM", title_number="AB", vin="BAD", vehicle_year=1700)
        errors = record.validate()
        self.assertEqual(3, len(errors))

    def test_to_dict_round_trips_through_from_dict(self) -> None:
        original = TitleRecord(
            id=42,
            state="NM",
            title_number="ABC1234",
            vin=_GOOD_VIN,
            vehicle_year=2003,
            make="HONDA",
            model="ACCORD",
            provider_id="PROV01",
            is_validated=True,
            validation_errors=[],
            created_at="2024-01-01T00:00:00",
        )
        restored = TitleRecord.from_dict(original.to_dict())
        self.assertEqual(original.id, restored.id)
        self.assertEqual(original.state, restored.state)
        self.assertEqual(original.vin, restored.vin)
        self.assertEqual(original.vehicle_year, restored.vehicle_year)
        self.assertEqual(original.make, restored.make)
        self.assertEqual(original.provider_id, restored.provider_id)
        self.assertEqual(original.is_validated, restored.is_validated)

    def test_from_dict_parses_pipe_separated_validation_errors(self) -> None:
        record = TitleRecord.from_dict({
            "state": "NM", "title_number": "ABC", "vin": _GOOD_VIN,
            "vehicle_year": 2003,
            "validation_errors": "VIN must be exactly 17 characters | Title number too short",
        })
        self.assertEqual(2, len(record.validation_errors))

    def test_from_dict_handles_list_validation_errors(self) -> None:
        record = TitleRecord.from_dict({
            "state": "NM", "title_number": "ABC", "vin": _GOOD_VIN,
            "vehicle_year": 2003,
            "validation_errors": ["error one", "error two"],
        })
        self.assertEqual(2, len(record.validation_errors))

    def test_from_dict_treats_is_validated_integer(self) -> None:
        record_true = TitleRecord.from_dict({"vehicle_year": 2003, "is_validated": 1})
        record_false = TitleRecord.from_dict({"vehicle_year": 2003, "is_validated": 0})
        self.assertTrue(record_true.is_validated)
        self.assertFalse(record_false.is_validated)

    def test_to_dict_contains_all_expected_keys(self) -> None:
        record = TitleRecord()
        d = record.to_dict()
        for f in ALL_FIELDS:
            self.assertIn(f, d)
        for f in OPERATIONAL_FIELDS:
            self.assertIn(f, d)
        for meta in ("id", "ocr_text", "is_validated", "validation_errors", "created_at"):
            self.assertIn(meta, d)

    def test_validation_errors_default_factory_is_independent(self) -> None:
        r1 = TitleRecord()
        r2 = TitleRecord()
        r1.validation_errors.append("err")
        self.assertEqual([], r2.validation_errors)

    def test_from_dict_casts_purchase_price_to_float(self) -> None:
        from decimal import Decimal
        record = TitleRecord.from_dict({"vehicle_year": 2003, "purchase_price": Decimal("12500.00")})
        self.assertIsInstance(record.purchase_price, float)
        self.assertAlmostEqual(12500.0, record.purchase_price)


class CorrectionModelTests(unittest.TestCase):
    def test_default_instance(self) -> None:
        c = Correction()
        self.assertEqual("", c.field_name)
        self.assertFalse(c.is_ground_truth)
        self.assertIsNone(c.source_file_path)

    def test_round_trip(self) -> None:
        c = Correction(
            id=1, record_id=42, field_name="vin",
            original_value="BAD", corrected_value=_GOOD_VIN, is_ground_truth=True,
        )
        restored = Correction.from_dict(c.to_dict())
        self.assertEqual(c.id, restored.id)
        self.assertEqual(c.corrected_value, restored.corrected_value)
        self.assertTrue(restored.is_ground_truth)

    def test_source_file_path_round_trips(self) -> None:
        c = Correction(id=2, record_id=7, field_name="make", source_file_path="/app/data/abc.png")
        restored = Correction.from_dict(c.to_dict())
        self.assertEqual("/app/data/abc.png", restored.source_file_path)

    def test_to_dict_includes_source_file_path(self) -> None:
        c = Correction(source_file_path="/some/path.jpg")
        self.assertIn("source_file_path", c.to_dict())

    def test_image_id_defaults_to_none(self) -> None:
        c = Correction()
        self.assertIsNone(c.image_id)

    def test_image_id_round_trips(self) -> None:
        c = Correction(id=3, record_id=8, field_name="vin", image_id=99)
        restored = Correction.from_dict(c.to_dict())
        self.assertEqual(99, restored.image_id)

    def test_to_dict_includes_image_id(self) -> None:
        c = Correction(image_id=42)
        self.assertIn("image_id", c.to_dict())
        self.assertEqual(42, c.to_dict()["image_id"])


class TitleImageModelTests(unittest.TestCase):
    def test_default_instance(self) -> None:
        img = TitleImage()
        self.assertIsNone(img.id)
        self.assertIsNone(img.title_record_id)
        self.assertEqual("", img.file_path)
        self.assertIsNone(img.file_hash)
        self.assertIsNone(img.mime_type)
        self.assertIsNone(img.original_filename)
        self.assertEqual("", img.created_at)

    def test_round_trip(self) -> None:
        img = TitleImage(
            id=5, title_record_id=10,
            file_path="/app/data/uploads/abc.png",
            file_hash="deadbeef" * 8,
            mime_type="image/png",
            original_filename="title.png",
            created_at="2024-05-01T12:00:00",
        )
        restored = TitleImage.from_dict(img.to_dict())
        self.assertEqual(img.id, restored.id)
        self.assertEqual(img.title_record_id, restored.title_record_id)
        self.assertEqual(img.file_path, restored.file_path)
        self.assertEqual(img.file_hash, restored.file_hash)
        self.assertEqual(img.mime_type, restored.mime_type)
        self.assertEqual(img.original_filename, restored.original_filename)
        self.assertEqual(img.created_at, restored.created_at)

    def test_to_dict_includes_all_keys(self) -> None:
        img = TitleImage(title_record_id=1, file_path="/p.png")
        d = img.to_dict()
        for key in ("id", "title_record_id", "file_path", "file_hash",
                    "mime_type", "original_filename", "created_at"):
            self.assertIn(key, d)

    def test_from_dict_handles_missing_optional_fields(self) -> None:
        img = TitleImage.from_dict({"title_record_id": 3, "file_path": "/f.png"})
        self.assertIsNone(img.id)
        self.assertIsNone(img.file_hash)
        self.assertIsNone(img.mime_type)
        self.assertEqual("", img.created_at)

    def test_from_dict_preserves_file_hash(self) -> None:
        sha = "a" * 64
        img = TitleImage.from_dict({"file_path": "/x.png", "file_hash": sha})
        self.assertEqual(sha, img.file_hash)


class TitleBackRecordModelTests(unittest.TestCase):
    def test_default_instance(self) -> None:
        b = TitleBackRecord()
        self.assertIsNone(b.title_record_id)
        self.assertIsNone(b.sale_price)

    def test_round_trip(self) -> None:
        b = TitleBackRecord(
            id=5, title_record_id=10, odometer_at_sale=95000,
            buyer_name="BUYER CORP", sale_price=5000.0, sale_date="2024-06-01",
        )
        restored = TitleBackRecord.from_dict(b.to_dict())
        self.assertEqual(b.buyer_name, restored.buyer_name)
        self.assertAlmostEqual(b.sale_price, restored.sale_price)

    def test_from_dict_casts_sale_price_decimal(self) -> None:
        from decimal import Decimal
        b = TitleBackRecord.from_dict({"sale_price": Decimal("3500.50")})
        self.assertIsInstance(b.sale_price, float)


class TrainingRunModelTests(unittest.TestCase):
    def test_default_instance(self) -> None:
        run = TrainingRun()
        self.assertEqual(0, run.sample_count)
        self.assertIsNone(run.notes)

    def test_round_trip(self) -> None:
        run = TrainingRun(id=3, exported_at="2024-01-01", sample_count=10, notes="test")
        restored = TrainingRun.from_dict(run.to_dict())
        self.assertEqual(run.sample_count, restored.sample_count)
        self.assertEqual(run.notes, restored.notes)


if __name__ == "__main__":
    unittest.main()
