import base64
import io
import json
import unittest
from unittest import mock

from PIL import Image

from ai_extraction import (
    _call_ollama,
    _crop_field_regions,
    _get_image_from_bytes,
    _image_to_base64_png,
    _merge_page_results,
    _normalize_image,
    _parse_ai_response,
    extract_fields_with_ai,
)

_GOOD_VIN = "1HGCM82633A004352"


def _make_small_rgb_image(width: int = 100, height: int = 80) -> Image.Image:
    return Image.new("RGB", (width, height), color=(200, 200, 200))


def _make_png_bytes(width: int = 40, height: int = 30) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


class NormalizeImageTests(unittest.TestCase):
    def test_resizes_image_larger_than_max_dimension(self) -> None:
        large = _make_small_rgb_image(2400, 1800)
        result = _normalize_image(large, max_dimension=1200)
        self.assertLessEqual(max(result.size), 1200)

    def test_does_not_resize_image_within_max_dimension(self) -> None:
        small = _make_small_rgb_image(800, 600)
        result = _normalize_image(small, max_dimension=1200)
        self.assertEqual((800, 600), result.size)

    def test_converts_rgba_to_rgb(self) -> None:
        rgba = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        result = _normalize_image(rgba)
        self.assertEqual("RGB", result.mode)

    def test_converts_palette_image_to_rgb(self) -> None:
        palette = Image.new("P", (50, 50))
        result = _normalize_image(palette)
        self.assertIn(result.mode, ("RGB", "L"))

    def test_returns_pil_image(self) -> None:
        img = _make_small_rgb_image()
        result = _normalize_image(img)
        self.assertIsInstance(result, Image.Image)


class ImageToBase64Tests(unittest.TestCase):
    def test_produces_valid_base64_string(self) -> None:
        img = _make_small_rgb_image(20, 20)
        encoded = _image_to_base64_png(img)
        # Must decode without error
        decoded = base64.b64decode(encoded)
        # Decoded bytes should be a valid PNG
        self.assertTrue(decoded[:8] == b"\x89PNG\r\n\x1a\n")

    def test_round_trips_through_pil(self) -> None:
        img = _make_small_rgb_image(10, 10)
        encoded = _image_to_base64_png(img)
        decoded_bytes = base64.b64decode(encoded)
        reopened = Image.open(io.BytesIO(decoded_bytes))
        self.assertEqual((10, 10), reopened.size)


class GetImageFromBytesTests(unittest.TestCase):
    def test_opens_png_bytes(self) -> None:
        png_bytes = _make_png_bytes()
        img = _get_image_from_bytes(png_bytes)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual((40, 30), img.size)


class ParseAiResponseTests(unittest.TestCase):
    def _valid_raw(self, **overrides):
        base = {
            "state": "NM",
            "title_number": "ABC1234",
            "vin": _GOOD_VIN,
            "vehicle_year": 2003,
            "confidence": {
                "state": 0.9,
                "title_number": 0.85,
                "vin": 0.95,
                "vehicle_year": 0.9,
            },
        }
        base.update(overrides)
        return base

    def test_parses_valid_response(self) -> None:
        result = _parse_ai_response(self._valid_raw())
        self.assertIsNotNone(result)
        self.assertEqual("NM", result["state"])
        self.assertEqual("ABC1234", result["title_number"])
        self.assertEqual(_GOOD_VIN, result["vin"])
        self.assertEqual(2003, result["vehicle_year"])
        self.assertAlmostEqual(0.9, result["confidence"]["state"])

    def test_returns_none_for_none_input(self) -> None:
        self.assertIsNone(_parse_ai_response(None))

    def test_returns_none_for_non_dict_input(self) -> None:
        self.assertIsNone(_parse_ai_response("not a dict"))
        self.assertIsNone(_parse_ai_response([1, 2, 3]))

    def test_normalises_state_to_uppercase_two_chars(self) -> None:
        result = _parse_ai_response(self._valid_raw(state="nm"))
        self.assertEqual("NM", result["state"])

    def test_strips_invalid_chars_from_state(self) -> None:
        result = _parse_ai_response(self._valid_raw(state="N.M."))
        self.assertEqual("NM", result["state"])

    def test_sets_state_none_for_non_string(self) -> None:
        result = _parse_ai_response(self._valid_raw(state=42))
        self.assertIsNone(result["state"])

    def test_normalises_title_number(self) -> None:
        result = _parse_ai_response(self._valid_raw(title_number="abc-1234"))
        self.assertEqual("ABC1234", result["title_number"])

    def test_sets_title_number_none_when_empty_after_strip(self) -> None:
        result = _parse_ai_response(self._valid_raw(title_number="---"))
        self.assertIsNone(result["title_number"])

    def test_rejects_vin_not_17_chars(self) -> None:
        result = _parse_ai_response(self._valid_raw(vin="SHORT"))
        self.assertIsNone(result["vin"])

    def test_strips_invalid_vin_characters_leaving_wrong_length(self) -> None:
        # "1HGCM82633A00435I" is 17 chars; after stripping invalid 'I' it
        # becomes 16 chars → must be rejected.
        result = _parse_ai_response(self._valid_raw(vin="1HGCM82633A00435I"))
        self.assertIsNone(result["vin"])

    def test_rejects_vehicle_year_out_of_range(self) -> None:
        result = _parse_ai_response(self._valid_raw(vehicle_year=1800))
        self.assertIsNone(result["vehicle_year"])
        result2 = _parse_ai_response(self._valid_raw(vehicle_year=2200))
        self.assertIsNone(result2["vehicle_year"])

    def test_clamps_confidence_to_0_1(self) -> None:
        raw = self._valid_raw()
        raw["confidence"]["state"] = 1.5
        raw["confidence"]["vin"] = -0.1
        result = _parse_ai_response(raw)
        self.assertEqual(1.0, result["confidence"]["state"])
        self.assertEqual(0.0, result["confidence"]["vin"])

    def test_handles_missing_confidence_section(self) -> None:
        raw = self._valid_raw()
        del raw["confidence"]
        result = _parse_ai_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(0.0, result["confidence"]["state"])

    def test_handles_non_dict_confidence_section(self) -> None:
        raw = self._valid_raw(confidence="bad")
        result = _parse_ai_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(0.0, result["confidence"]["vin"])


class MergePageResultsTests(unittest.TestCase):
    def _result(self, **kwargs):
        defaults = {
            "state": None,
            "title_number": None,
            "vin": None,
            "vehicle_year": None,
            "confidence": {"state": 0.0, "title_number": 0.0, "vin": 0.0, "vehicle_year": 0.0},
        }
        defaults.update(kwargs)
        return defaults

    def test_single_result_is_returned_unchanged(self) -> None:
        r = self._result(
            vin=_GOOD_VIN,
            confidence={"state": 0.0, "title_number": 0.0, "vin": 0.9, "vehicle_year": 0.0},
        )
        merged = _merge_page_results([r])
        self.assertEqual(_GOOD_VIN, merged["vin"])
        self.assertAlmostEqual(0.9, merged["confidence"]["vin"])

    def test_prefers_higher_confidence_value(self) -> None:
        r1 = self._result(
            state="NM",
            confidence={"state": 0.6, "title_number": 0.0, "vin": 0.0, "vehicle_year": 0.0},
        )
        r2 = self._result(
            state="TX",
            confidence={"state": 0.9, "title_number": 0.0, "vin": 0.0, "vehicle_year": 0.0},
        )
        merged = _merge_page_results([r1, r2])
        self.assertEqual("TX", merged["state"])
        self.assertAlmostEqual(0.9, merged["confidence"]["state"])

    def test_none_value_does_not_overwrite_existing(self) -> None:
        r1 = self._result(
            vin=_GOOD_VIN,
            confidence={"state": 0.0, "title_number": 0.0, "vin": 0.8, "vehicle_year": 0.0},
        )
        r2 = self._result(
            vin=None,
            confidence={"state": 0.0, "title_number": 0.0, "vin": 0.0, "vehicle_year": 0.0},
        )
        merged = _merge_page_results([r1, r2])
        self.assertEqual(_GOOD_VIN, merged["vin"])

    def test_empty_list_returns_all_none(self) -> None:
        merged = _merge_page_results([])
        for field in ("state", "title_number", "vin", "vehicle_year"):
            self.assertIsNone(merged[field])
            self.assertEqual(0.0, merged["confidence"][field])


class CallOllamaTests(unittest.TestCase):
    @mock.patch("ai_extraction.requests.post")
    def test_returns_parsed_json_on_success(self, post_mock: mock.Mock) -> None:
        response_body = {"state": "NM", "vin": _GOOD_VIN}
        response_mock = mock.Mock()
        response_mock.raise_for_status = mock.Mock()
        response_mock.json.return_value = {"response": json.dumps(response_body)}
        post_mock.return_value = response_mock

        result = _call_ollama("b64data", "http://localhost:11434", "moondream", 30)
        self.assertEqual(response_body, result)
        post_mock.assert_called_once()

    @mock.patch("ai_extraction.requests.post")
    def test_returns_none_on_request_exception(self, post_mock: mock.Mock) -> None:
        import requests as req_lib

        post_mock.side_effect = req_lib.ConnectionError("timeout")
        result = _call_ollama("b64data", "http://localhost:11434", "moondream", 5)
        self.assertIsNone(result)

    @mock.patch("ai_extraction.requests.post")
    def test_returns_none_on_json_decode_error(self, post_mock: mock.Mock) -> None:
        response_mock = mock.Mock()
        response_mock.raise_for_status = mock.Mock()
        response_mock.json.return_value = {"response": "not valid json {{{"}
        post_mock.return_value = response_mock

        result = _call_ollama("b64data", "http://localhost:11434", "moondream", 5)
        self.assertIsNone(result)

    @mock.patch("ai_extraction.requests.post")
    def test_returns_none_on_http_error(self, post_mock: mock.Mock) -> None:
        import requests as req_lib

        response_mock = mock.Mock()
        response_mock.raise_for_status.side_effect = req_lib.HTTPError("503")
        post_mock.return_value = response_mock

        result = _call_ollama("b64data", "http://localhost:11434", "moondream", 5)
        self.assertIsNone(result)


class ExtractFieldsWithAiTests(unittest.TestCase):
    def _ai_result(self):
        return {
            "state": "NM",
            "title_number": "ABC1234",
            "vin": _GOOD_VIN,
            "vehicle_year": 2003,
            "confidence": {
                "state": 0.9,
                "title_number": 0.85,
                "vin": 0.95,
                "vehicle_year": 0.9,
            },
        }

    @mock.patch("ai_extraction._call_ollama", return_value=None)
    @mock.patch("ai_extraction._normalize_image")
    @mock.patch("ai_extraction._image_to_base64_png", return_value="b64")
    @mock.patch("ai_extraction._get_image_from_bytes")
    def test_returns_none_when_all_calls_fail(
        self,
        get_image_mock: mock.Mock,
        _b64_mock: mock.Mock,
        _norm_mock: mock.Mock,
        _call_mock: mock.Mock,
    ) -> None:
        get_image_mock.return_value = _make_small_rgb_image()
        result = extract_fields_with_ai("title.png", _make_png_bytes())
        self.assertIsNone(result)

    @mock.patch(
        "ai_extraction._call_ollama",
        return_value={
            "state": "NM",
            "title_number": "ABC1234",
            "vin": _GOOD_VIN,
            "vehicle_year": 2003,
            "confidence": {
                "state": 0.9,
                "title_number": 0.85,
                "vin": 0.95,
                "vehicle_year": 0.9,
            },
        },
    )
    @mock.patch("ai_extraction._normalize_image", side_effect=lambda img, **kw: img)
    @mock.patch("ai_extraction._image_to_base64_png", return_value="b64data")
    @mock.patch("ai_extraction._get_image_from_bytes")
    def test_returns_merged_result_on_success(
        self,
        get_image_mock: mock.Mock,
        _b64_mock: mock.Mock,
        _norm_mock: mock.Mock,
        _call_mock: mock.Mock,
    ) -> None:
        get_image_mock.return_value = _make_small_rgb_image()
        result = extract_fields_with_ai("title.png", _make_png_bytes())
        self.assertIsNotNone(result)
        self.assertEqual("NM", result["state"])
        self.assertEqual(_GOOD_VIN, result["vin"])
        self.assertAlmostEqual(0.9, result["confidence"]["state"])

    @mock.patch("ai_extraction._call_ollama", return_value=None)
    @mock.patch("ai_extraction._normalize_image", side_effect=lambda img, **kw: img)
    @mock.patch("ai_extraction._image_to_base64_png", return_value="b64data")
    @mock.patch("ai_extraction._get_page_images_from_pdf")
    def test_processes_pdf_pages(
        self,
        get_pages_mock: mock.Mock,
        _b64_mock: mock.Mock,
        _norm_mock: mock.Mock,
        _call_mock: mock.Mock,
    ) -> None:
        page_images = [_make_small_rgb_image(), _make_small_rgb_image()]
        get_pages_mock.return_value = page_images
        result = extract_fields_with_ai("document.pdf", b"fake-pdf-bytes")
        # All calls failed → None
        self.assertIsNone(result)
        # Each page: 1 full-image call + 2 crop calls (from _crop_field_regions).
        crops_per_page = len(_crop_field_regions(_make_small_rgb_image()))
        expected_calls = len(page_images) * (1 + crops_per_page)
        self.assertEqual(expected_calls, _call_mock.call_count)

    @mock.patch("ai_extraction._call_ollama", return_value=None)
    @mock.patch("ai_extraction._normalize_image", side_effect=lambda img, **kw: img)
    @mock.patch("ai_extraction._image_to_base64_png", return_value="b64data")
    @mock.patch("ai_extraction._get_image_from_bytes")
    def test_uses_env_for_endpoint_and_model(
        self,
        get_image_mock: mock.Mock,
        _b64_mock: mock.Mock,
        _norm_mock: mock.Mock,
        call_mock: mock.Mock,
    ) -> None:
        get_image_mock.return_value = _make_small_rgb_image()
        with mock.patch.dict(
            "os.environ",
            {"AI_ENDPOINT": "http://myollama:9999", "AI_MODEL": "llava", "AI_TIMEOUT": "15"},
        ):
            extract_fields_with_ai("title.png", _make_png_bytes())
        call_args = call_mock.call_args
        self.assertEqual("http://myollama:9999", call_args[0][1])
        self.assertEqual("llava", call_args[0][2])
        self.assertEqual(15, call_args[0][3])


class CropFieldRegionsTests(unittest.TestCase):
    def test_returns_two_crops(self) -> None:
        image = _make_small_rgb_image(400, 300)
        crops = _crop_field_regions(image)
        self.assertEqual(2, len(crops))
        for crop in crops:
            self.assertIsInstance(crop, Image.Image)

    def test_top_strip_spans_full_width(self) -> None:
        image = _make_small_rgb_image(400, 300)
        crops = _crop_field_regions(image)
        top_strip = crops[0]
        self.assertEqual(400, top_strip.width)
        # ~12 % of 300 = 36 px
        self.assertAlmostEqual(top_strip.height, int(300 * 0.12), delta=1)

    def test_top_right_crop_is_narrower_than_full_image(self) -> None:
        image = _make_small_rgb_image(400, 300)
        crops = _crop_field_regions(image)
        top_right = crops[1]
        # ~15 % of 400 = 60 px wide
        self.assertLess(top_right.width, 400)
        self.assertAlmostEqual(top_right.width, int(400 * 0.15), delta=1)

    def test_handles_small_image_without_error(self) -> None:
        tiny = _make_small_rgb_image(10, 10)
        crops = _crop_field_regions(tiny)
        self.assertEqual(2, len(crops))
        for crop in crops:
            self.assertGreaterEqual(crop.width, 1)
            self.assertGreaterEqual(crop.height, 1)

    def test_crops_are_subregions_of_original(self) -> None:
        """Each crop must fit entirely within the original image dimensions."""
        image = _make_small_rgb_image(800, 600)
        for crop in _crop_field_regions(image):
            self.assertLessEqual(crop.width, 800)
            self.assertLessEqual(crop.height, 600)


class ExtractionSmokeTest(unittest.TestCase):
    """End-to-end smoke test using a synthetic title-like image.

    Verifies that the extraction pipeline (normalisation → base64 encoding →
    AI call → response parsing → merge) correctly propagates a known result
    without contacting a real Ollama endpoint.
    """

    _KNOWN_VIN = "1GNDT13S372145342"
    _KNOWN_RESULT = {
        "state": "MD",
        "title_number": "58312391",
        "vin": _KNOWN_VIN,
        "vehicle_year": 2007,
        "confidence": {
            "state": 0.9,
            "title_number": 0.95,
            "vin": 0.95,
            "vehicle_year": 0.9,
        },
    }

    def _make_title_image_bytes(self) -> bytes:
        """Return PNG bytes for a minimal synthetic vehicle title image."""
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), color=(240, 240, 230)).save(buf, format="PNG")
        return buf.getvalue()

    @mock.patch("ai_extraction._call_ollama")
    def test_known_result_propagates_through_pipeline(
        self, call_mock: mock.Mock
    ) -> None:
        call_mock.return_value = self._KNOWN_RESULT
        result = extract_fields_with_ai("title.png", self._make_title_image_bytes())
        self.assertIsNotNone(result)
        self.assertEqual("MD", result["state"])
        self.assertEqual("58312391", result["title_number"])
        self.assertEqual(self._KNOWN_VIN, result["vin"])
        self.assertEqual(2007, result["vehicle_year"])

    @mock.patch("ai_extraction._call_ollama")
    def test_crop_results_merged_with_full_image_result(
        self, call_mock: mock.Mock
    ) -> None:
        """High-confidence crop result should beat low-confidence full-page result."""
        low_conf_result = {
            "state": "XX",
            "title_number": "WRONG",
            "vin": None,
            "vehicle_year": None,
            "confidence": {
                "state": 0.4,
                "title_number": 0.3,
                "vin": 0.0,
                "vehicle_year": 0.0,
            },
        }
        high_conf_crop = {
            "state": "MD",
            "title_number": "58312391",
            "vin": self._KNOWN_VIN,
            "vehicle_year": 2007,
            "confidence": {
                "state": 0.9,
                "title_number": 0.95,
                "vin": 0.95,
                "vehicle_year": 0.9,
            },
        }
        # First call = full page (low confidence), subsequent calls = crops.
        call_mock.side_effect = [low_conf_result, high_conf_crop, high_conf_crop]
        result = extract_fields_with_ai("title.png", self._make_title_image_bytes())
        self.assertIsNotNone(result)
        # Crop values should win because they carry higher confidence.
        self.assertEqual("MD", result["state"])
        self.assertEqual("58312391", result["title_number"])
        self.assertEqual(self._KNOWN_VIN, result["vin"])
        self.assertEqual(2007, result["vehicle_year"])


if __name__ == "__main__":
    unittest.main()
