import json
import unittest
from io import BytesIO
from unittest import mock

from PIL import Image as _PILImage

from web_app import (
    _apply_word_confidence,
    _build_llm_modelfile,
    _check_ai_status,
    _docker_client,
    _extract_text_from_image,
    _extract_text_from_pdf,
    _get_ai_pull_progress,
    _get_service_containers,
    _get_service_logs,
    _get_word_confidences,
    _parse_pull_progress_from_logs,
    _run_extraction,
    _start_ai_services,
    _tesseract_fields_and_confidence,
    allowed_file,
    create_app,
    extract_text_from_upload,
    get_port_from_environment,
    parse_extracted_fields,
)
from ai_extraction import _ALL_FIELDS

_GOOD_VIN = "1HGCM82633A004352"
_GOOD_OCR = f"VIN {_GOOD_VIN} YEAR 2003 TITLE ABC1234"
_GOOD_EXTRACTION = {
    "fields": {
        "state": None,
        "title_number": "ABC1234",
        "vin": _GOOD_VIN,
        "vehicle_year": 2003,
        **{f: None for f in (
            "make", "model", "body_style", "color", "odometer",
            "owner_name", "owner_address", "purchase_price", "sale_date", "issue_date",
        )},
    },
    "raw_text": _GOOD_OCR,
    "source": "tesseract",
    "confidence": {
        "state": 0.0,
        "title_number": 0.7,
        "vin": 0.7,
        "vehicle_year": 0.7,
        **{f: 0.0 for f in (
            "make", "model", "body_style", "color", "odometer",
            "owner_name", "owner_address", "purchase_price", "sale_date", "issue_date",
        )},
    },
    "low_confidence": ["state", "make", "model", "body_style", "color", "odometer",
                       "owner_name", "owner_address", "purchase_price", "sale_date", "issue_date"],
}


class WebAppParsingTests(unittest.TestCase):
    def test_allowed_file_supports_pdf_and_images(self) -> None:
        self.assertTrue(allowed_file("title.pdf"))
        self.assertTrue(allowed_file("title.jpeg"))
        self.assertFalse(allowed_file("title.txt"))

    def test_parse_extracted_fields_reads_labeled_values(self) -> None:
        text = """
        STATE: NM
        TITLE NUMBER: ABC-1234
        VIN: 1HGCM82633A004352
        YEAR: 2003
        """
        fields = parse_extracted_fields(text)
        self.assertEqual("NM", fields["state"])
        self.assertEqual("ABC1234", fields["title_number"])
        self.assertEqual("1HGCM82633A004352", fields["vin"])
        self.assertEqual(2003, fields["vehicle_year"])

    def test_parse_extracted_fields_uses_fallback_title(self) -> None:
        text = "VIN 1HGCM82633A004352 YEAR 2003 REF ZX-90077"
        fields = parse_extracted_fields(text)
        self.assertEqual("ZX90077", fields["title_number"])

    def test_parse_extracted_fields_finds_8digit_title_number(self) -> None:
        # 8-digit numeric title numbers are common; they should be found without
        # a TITLE label when no labeled value is present.
        text = f"VIN {_GOOD_VIN} YEAR 2007 58312391"
        fields = parse_extracted_fields(text)
        self.assertEqual("58312391", fields["title_number"])

    def test_parse_extracted_fields_corrects_ocr_noise_in_8digit_title(self) -> None:
        # O→0 and I→1 are common OCR misreads in numeric title numbers.
        text = f"VIN {_GOOD_VIN} YEAR 2007 5831239I"
        fields = parse_extracted_fields(text)
        self.assertEqual("58312391", fields["title_number"])

    def test_parse_extracted_fields_8digit_does_not_override_labeled_title(self) -> None:
        # An explicitly labeled TITLE NUMBER should always win.
        text = f"TITLE NUMBER: ABC1234 58312391 VIN {_GOOD_VIN} YEAR 2007"
        fields = parse_extracted_fields(text)
        self.assertEqual("ABC1234", fields["title_number"])

    @mock.patch("web_app._extract_text_from_pdf", return_value="pdf text")
    @mock.patch("web_app._extract_text_from_image", return_value="image text")
    def test_extract_text_from_upload_routes_by_extension(
        self,
        image_mock: mock.Mock,
        pdf_mock: mock.Mock,
    ) -> None:
        self.assertEqual("pdf text", extract_text_from_upload("sample.PDF", b"x"))
        self.assertEqual("image text", extract_text_from_upload("sample.png", b"x"))
        self.assertEqual("", extract_text_from_upload("sample", b"x"))
        pdf_mock.assert_called_once_with(b"x")
        image_mock.assert_called_once_with(b"x")

    @mock.patch.dict("os.environ", {"PORT": "5050"}, clear=True)
    def test_get_port_from_environment_reads_valid_port(self) -> None:
        self.assertEqual(5050, get_port_from_environment())

    @mock.patch.dict("os.environ", {"PORT": "invalid"}, clear=True)
    def test_get_port_from_environment_falls_back_for_invalid_value(self) -> None:
        self.assertEqual(8000, get_port_from_environment())

    @mock.patch.dict("os.environ", {"PORT": "70000"}, clear=True)
    def test_get_port_from_environment_falls_back_for_out_of_range_value(self) -> None:
        self.assertEqual(8000, get_port_from_environment())

    @mock.patch("web_app.pytesseract.image_to_string", return_value="ocr image text")
    @mock.patch("web_app.Image.open")
    def test_extract_text_from_image_uses_tesseract(
        self,
        image_open_mock: mock.Mock,
        image_to_string_mock: mock.Mock,
    ) -> None:
        fake_image = _PILImage.new("RGB", (100, 80), color=(200, 200, 200))
        image_open_mock.return_value = fake_image
        result = _extract_text_from_image(b"img-bytes")
        self.assertEqual("ocr image text", result)
        image_to_string_mock.assert_called_once()

    @mock.patch("web_app.pytesseract.image_to_string", return_value="ocr fallback text")
    @mock.patch("web_app.Image.open")
    @mock.patch("web_app.fitz.open")
    def test_extract_text_from_pdf_uses_native_text_and_ocr_fallback(
        self,
        fitz_open_mock: mock.Mock,
        image_open_mock: mock.Mock,
        image_to_string_mock: mock.Mock,
    ) -> None:
        page_with_text = mock.Mock()
        page_with_text.get_text.return_value = "native text"

        page_without_text = mock.Mock()
        page_without_text.get_text.return_value = "   "
        pixmap = mock.Mock()
        pixmap.tobytes.return_value = b"png-bytes"
        page_without_text.get_pixmap.return_value = pixmap

        document = mock.MagicMock()
        document.__iter__.return_value = [page_with_text, page_without_text]
        fitz_open_mock.return_value = document
        image_open_mock.return_value = _PILImage.new("RGB", (100, 80), color=(200, 200, 200))

        result = _extract_text_from_pdf(b"pdf-bytes")
        self.assertIn("native text", result)
        self.assertIn("ocr fallback text", result)
        fitz_open_mock.assert_called_once()
        image_to_string_mock.assert_called_once()

    def test_index_route_rejects_missing_file(self) -> None:
        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.post("/", data={}, content_type="multipart/form-data")
        self.assertIn(b"Please select a PDF or image file.", response.data)

    @mock.patch("web_app.insert_title_record")
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    @mock.patch("web_app._run_extraction", return_value=_GOOD_EXTRACTION)
    def test_index_route_processes_upload(
        self,
        extraction_mock: mock.Mock,
        create_connection_mock: mock.Mock,
        initialize_database_mock: mock.Mock,
        insert_mock: mock.Mock,
    ) -> None:
        connection = mock.Mock()
        create_connection_mock.return_value = connection
        insert_mock.return_value = {"id": 1, "is_validated": True, "validation_errors": []}

        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.post(
            "/",
            data={"file": (BytesIO(b"fake"), "title.pdf"), "state": "NM"},
            content_type="multipart/form-data",
        )

        self.assertEqual(200, response.status_code)
        self.assertIn(b"Record extracted, validated, and saved.", response.data)
        extraction_mock.assert_called_once()
        initialize_database_mock.assert_called_once_with(connection)
        insert_mock.assert_called_once()
        connection.close.assert_called_once()

    @mock.patch("web_app.insert_title_record")
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    @mock.patch("web_app._run_extraction", return_value=_GOOD_EXTRACTION)
    def test_index_route_includes_extraction_info_in_response(
        self,
        _extraction_mock: mock.Mock,
        create_connection_mock: mock.Mock,
        _initialize_mock: mock.Mock,
        insert_mock: mock.Mock,
    ) -> None:
        create_connection_mock.return_value = mock.Mock()
        insert_mock.return_value = {"id": 2, "is_validated": True, "validation_errors": []}

        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.post(
            "/",
            data={"file": (BytesIO(b"fake"), "title.png"), "state": "NM"},
            content_type="multipart/form-data",
        )

        self.assertEqual(200, response.status_code)
        self.assertIn(b"TESSERACT", response.data)
        self.assertIn(b"Extraction Details", response.data)

    @mock.patch("web_app._run_extraction", return_value=None)
    def test_index_route_shows_error_when_extraction_returns_none(
        self,
        _extraction_mock: mock.Mock,
    ) -> None:
        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.post(
            "/",
            data={"file": (BytesIO(b"fake"), "title.png")},
            content_type="multipart/form-data",
        )

        self.assertIn(b"No text could be extracted from the file.", response.data)


class RunExtractionTests(unittest.TestCase):
    """Tests for _run_extraction provider selection logic."""

    @mock.patch("web_app.extract_text_from_upload", return_value=_GOOD_OCR)
    def test_tesseract_provider_returns_parsed_fields(
        self, _ocr_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="tesseract")
        self.assertIsNotNone(result)
        self.assertEqual("tesseract", result["source"])
        self.assertEqual(_GOOD_VIN, result["fields"]["vin"])
        self.assertEqual(2003, result["fields"]["vehicle_year"])

    @mock.patch("web_app.extract_text_from_upload", return_value="")
    def test_tesseract_provider_returns_none_when_no_text(
        self, _ocr_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="tesseract")
        self.assertIsNone(result)

    @mock.patch(
        "web_app.extract_fields_with_ai",
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
    @mock.patch("web_app.extract_text_from_upload", return_value=_GOOD_OCR)
    def test_ai_provider_uses_ai_fields(
        self, _ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="ai")
        self.assertIsNotNone(result)
        self.assertEqual("ai", result["source"])
        self.assertEqual("NM", result["fields"]["state"])
        # Core fields should not be low-confidence.
        self.assertNotIn("state", result["low_confidence"])
        self.assertNotIn("vin", result["low_confidence"])

    @mock.patch(
        "web_app.extract_fields_with_ai",
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
    @mock.patch("web_app.extract_text_from_upload")
    def test_ai_provider_skips_tesseract_when_ai_succeeds(
        self, ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="ai")
        self.assertIsNotNone(result)
        self.assertEqual("ai", result["source"])
        self.assertEqual("", result["raw_text"])
        ocr_mock.assert_not_called()

    @mock.patch("web_app.extract_fields_with_ai", return_value=None)
    @mock.patch("web_app.extract_text_from_upload", return_value=_GOOD_OCR)
    def test_ai_provider_falls_back_to_tesseract_when_ai_unavailable(
        self, _ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="ai")
        self.assertIsNotNone(result)
        self.assertEqual("tesseract", result["source"])
        self.assertEqual(_GOOD_VIN, result["fields"]["vin"])

    @mock.patch("web_app.extract_fields_with_ai", return_value=None)
    @mock.patch("web_app.extract_text_from_upload", return_value="")
    def test_extraction_returns_none_when_ai_and_text_both_missing(
        self, _ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="hybrid")
        self.assertIsNone(result)

    @mock.patch(
        "web_app.extract_fields_with_ai",
        return_value={
            "state": "TX",
            "title_number": "XYZ999",
            "vin": _GOOD_VIN,
            "vehicle_year": 2020,
            "confidence": {
                "state": 0.9,
                "title_number": 0.8,
                "vin": 0.95,
                "vehicle_year": 0.9,
            },
        },
    )
    @mock.patch("web_app.extract_text_from_upload", return_value=_GOOD_OCR)
    def test_hybrid_provider_uses_ai_for_high_confidence_fields(
        self, _ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="hybrid", threshold=0.6)
        self.assertIsNotNone(result)
        self.assertIn(result["source"], ("ai", "hybrid"))
        self.assertEqual("TX", result["fields"]["state"])
        self.assertEqual(_GOOD_VIN, result["fields"]["vin"])

    @mock.patch(
        "web_app.extract_fields_with_ai",
        return_value={
            "state": "TX",
            "title_number": "XYZ999",
            "vin": _GOOD_VIN,
            "vehicle_year": 2020,
            "confidence": {
                "state": 0.9,
                "title_number": 0.8,
                "vin": 0.95,
                "vehicle_year": 0.9,
            },
        },
    )
    @mock.patch("web_app.extract_text_from_upload")
    def test_hybrid_provider_skips_tesseract_when_ai_confident_for_all_fields(
        self, ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="hybrid", threshold=0.6)
        self.assertIsNotNone(result)
        self.assertEqual("ai", result["source"])
        self.assertEqual("", result["raw_text"])
        ocr_mock.assert_not_called()

    @mock.patch(
        "web_app.extract_fields_with_ai",
        return_value={
            "state": None,
            "title_number": None,
            "vin": None,
            "vehicle_year": None,
            "confidence": {
                "state": 0.0,
                "title_number": 0.0,
                "vin": 0.0,
                "vehicle_year": 0.0,
            },
        },
    )
    @mock.patch("web_app.extract_text_from_upload", return_value=_GOOD_OCR)
    def test_hybrid_provider_falls_back_to_tesseract_for_all_low_confidence(
        self, _ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="hybrid", threshold=0.6)
        self.assertIsNotNone(result)
        self.assertEqual("tesseract", result["source"])
        self.assertEqual(_GOOD_VIN, result["fields"]["vin"])

    @mock.patch("web_app.extract_fields_with_ai", side_effect=RuntimeError("boom"))
    @mock.patch("web_app.extract_text_from_upload", return_value=_GOOD_OCR)
    def test_hybrid_provider_handles_ai_exception(
        self, _ocr_mock: mock.Mock, _ai_mock: mock.Mock
    ) -> None:
        result = _run_extraction("title.png", b"bytes", provider="hybrid")
        self.assertIsNotNone(result)
        self.assertEqual("tesseract", result["source"])

    def test_tesseract_fields_and_confidence_assigns_nominal_confidence(self) -> None:
        fields, conf = _tesseract_fields_and_confidence(_GOOD_OCR)
        self.assertEqual(_GOOD_VIN, fields["vin"])
        self.assertEqual(0.7, conf["vin"])
        self.assertEqual(0.0, conf["state"])


class WordConfidenceTests(unittest.TestCase):
    def test_get_word_confidences_returns_empty_dict_on_non_image(self) -> None:
        result = _get_word_confidences(b"not-an-image")
        self.assertIsInstance(result, dict)
        self.assertEqual({}, result)

    def test_apply_word_confidence_overrides_nominal_when_word_found(self) -> None:
        fields = {"state": None, "title_number": "ABC123", "vin": _GOOD_VIN, "vehicle_year": 2003}
        confidence = {"state": 0.0, "title_number": 0.7, "vin": 0.7, "vehicle_year": 0.7}
        word_conf = {"ABC123": 0.92, "2003": 0.88}
        _apply_word_confidence(fields, confidence, word_conf)
        self.assertAlmostEqual(0.92, confidence["title_number"])
        self.assertAlmostEqual(0.88, confidence["vehicle_year"])
        # VIN not in word_conf → stays at nominal 0.7
        self.assertAlmostEqual(0.7, confidence["vin"])
        # state is None → stays at 0.0
        self.assertAlmostEqual(0.0, confidence["state"])

    def test_apply_word_confidence_ignores_none_values(self) -> None:
        fields = {"state": None, "title_number": None, "vin": None, "vehicle_year": None}
        confidence = {"state": 0.0, "title_number": 0.0, "vin": 0.0, "vehicle_year": 0.0}
        word_conf = {"SOMETHING": 0.9}
        _apply_word_confidence(fields, confidence, word_conf)
        for f in confidence:
            self.assertAlmostEqual(0.0, confidence[f])

    def test_apply_word_confidence_does_not_raise_on_empty_word_conf(self) -> None:
        fields = {"state": "NM", "title_number": "ABC123", "vin": _GOOD_VIN, "vehicle_year": 2003}
        confidence = {"state": 0.7, "title_number": 0.7, "vin": 0.7, "vehicle_year": 0.7}
        _apply_word_confidence(fields, confidence, {})
        # Nothing changed
        for f in confidence:
            self.assertAlmostEqual(0.7, confidence[f])


class HealthAndStatusEndpointTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.get("/health")

        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertEqual("ok", data["status"])

    @mock.patch("web_app._check_ai_status", return_value="unavailable")
    def test_api_status_returns_provider_and_ai_status(self, _mock: mock.Mock) -> None:
        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.get("/api/status")

        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertIn("extraction_provider", data)
        self.assertIn("ai_status", data)
        self.assertEqual("unavailable", data["ai_status"])

    @mock.patch("web_app._check_ai_status", return_value="ready")
    def test_api_status_returns_ready_when_ai_reachable(self, _mock: mock.Mock) -> None:
        app = create_app()
        app.testing = True
        client = app.test_client()

        response = client.get("/api/status")

        self.assertEqual(200, response.status_code)
        self.assertEqual("ready", json.loads(response.data)["ai_status"])

    @mock.patch("web_app.EXTRACTION_PROVIDER", "tesseract")
    @mock.patch("web_app._AI_STATUS_CACHE", {"status": "unknown", "checked_at": 0.0})
    def test_check_ai_status_returns_not_configured_for_tesseract(self) -> None:
        import web_app as _wa

        status = _check_ai_status()
        self.assertEqual("not_configured", status)
        self.assertEqual("not_configured", _wa._AI_STATUS_CACHE["status"])
        self.assertGreater(float(_wa._AI_STATUS_CACHE["checked_at"]), 0.0)

    @mock.patch("web_app.EXTRACTION_PROVIDER", "hybrid")
    @mock.patch("web_app._AI_STATUS_CACHE", {"status": "unknown", "checked_at": 0.0})
    @mock.patch("web_app.requests.get")
    def test_check_ai_status_returns_ready_when_model_present(
        self, get_mock: mock.Mock
    ) -> None:
        import os
        import web_app as _wa

        model_name = os.getenv("AI_MODEL", "moondream")
        resp = mock.Mock()
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = {"models": [{"name": model_name}]}
        get_mock.return_value = resp

        status = _check_ai_status()
        self.assertEqual("ready", status)
        self.assertEqual("ready", _wa._AI_STATUS_CACHE["status"])

    @mock.patch("web_app.EXTRACTION_PROVIDER", "hybrid")
    @mock.patch("web_app._AI_STATUS_CACHE", {"status": "unknown", "checked_at": 0.0})
    @mock.patch("web_app.requests.get")
    def test_check_ai_status_returns_loading_when_model_not_yet_present(
        self, get_mock: mock.Mock
    ) -> None:
        import web_app as _wa

        resp = mock.Mock()
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = {"models": []}   # API up, model not downloaded yet
        get_mock.return_value = resp

        status = _check_ai_status()
        self.assertEqual("loading", status)
        self.assertEqual("loading", _wa._AI_STATUS_CACHE["status"])

    @mock.patch("web_app.EXTRACTION_PROVIDER", "hybrid")
    @mock.patch("web_app._AI_STATUS_CACHE", {"status": "unknown", "checked_at": 0.0})
    @mock.patch("web_app.requests.get", side_effect=ConnectionError("refused"))
    def test_check_ai_status_returns_unavailable_when_endpoint_unreachable(
        self, _get_mock: mock.Mock
    ) -> None:
        import web_app as _wa

        status = _check_ai_status()
        self.assertEqual("unavailable", status)
        self.assertEqual("unavailable", _wa._AI_STATUS_CACHE["status"])


class MaintenanceEndpointTests(unittest.TestCase):
    """Tests for /api/maintenance/* endpoints."""

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    # ── /api/maintenance/containers ────────────────────────────────────────

    @mock.patch("web_app._docker_client", return_value=None)
    def test_containers_returns_unavailable_when_no_socket(
        self, _mock: mock.Mock
    ) -> None:
        client = self._app().test_client()
        response = client.get("/api/maintenance/containers")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertFalse(data["docker_available"])
        self.assertIsNotNone(data["error"])

    @mock.patch("web_app._get_service_containers")
    @mock.patch("web_app._docker_client")
    def test_containers_returns_service_dict_when_docker_available(
        self, docker_mock: mock.Mock, containers_mock: mock.Mock
    ) -> None:
        docker_mock.return_value = mock.Mock()
        containers_mock.return_value = {
            "title-entry-tool": {"id": "abc", "name": "t", "state": "running", "health": "healthy"},
            "postgres": None,
            "ollama": None,
            "ollama-init": None,
        }
        client = self._app().test_client()
        response = client.get("/api/maintenance/containers")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertTrue(data["docker_available"])
        self.assertIn("title-entry-tool", data["services"])
        self.assertEqual("running", data["services"]["title-entry-tool"]["state"])

    # ── /api/maintenance/logs/<service> ────────────────────────────────────

    @mock.patch("web_app._docker_client", return_value=None)
    def test_logs_returns_503_when_no_socket(self, _mock: mock.Mock) -> None:
        client = self._app().test_client()
        response = client.get("/api/maintenance/logs/postgres")
        self.assertEqual(503, response.status_code)

    def test_logs_returns_400_for_unknown_service(self) -> None:
        client = self._app().test_client()
        response = client.get("/api/maintenance/logs/unknown-svc")
        self.assertEqual(400, response.status_code)

    @mock.patch("web_app._get_service_logs")
    @mock.patch("web_app._docker_client")
    def test_logs_returns_lines_for_known_service(
        self, docker_mock: mock.Mock, logs_mock: mock.Mock
    ) -> None:
        docker_mock.return_value = mock.Mock()
        logs_mock.return_value = {"lines": ["line1", "line2"], "error": None}
        client = self._app().test_client()
        response = client.get("/api/maintenance/logs/postgres?lines=50")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertEqual("postgres", data["service"])
        self.assertEqual(["line1", "line2"], data["lines"])
        self.assertIsNone(data["error"])

    # ── POST /api/maintenance/ai/start ────────────────────────────────────

    @mock.patch("web_app._docker_client", return_value=None)
    def test_start_ai_returns_503_when_no_socket(self, _mock: mock.Mock) -> None:
        client = self._app().test_client()
        response = client.post("/api/maintenance/ai/start")
        self.assertEqual(503, response.status_code)
        data = json.loads(response.data)
        self.assertFalse(data["success"])

    @mock.patch("web_app._start_ai_services")
    @mock.patch("web_app._docker_client")
    def test_start_ai_returns_200_on_success(
        self, docker_mock: mock.Mock, start_mock: mock.Mock
    ) -> None:
        docker_mock.return_value = mock.Mock()
        start_mock.return_value = {
            "success": True,
            "messages": ["'ollama' started."],
            "errors": [],
        }
        client = self._app().test_client()
        response = client.post("/api/maintenance/ai/start")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertTrue(data["success"])

    @mock.patch("web_app._start_ai_services")
    @mock.patch("web_app._docker_client")
    def test_start_ai_returns_500_on_failure(
        self, docker_mock: mock.Mock, start_mock: mock.Mock
    ) -> None:
        docker_mock.return_value = mock.Mock()
        start_mock.return_value = {
            "success": False,
            "messages": [],
            "errors": ["Container not found."],
        }
        client = self._app().test_client()
        response = client.post("/api/maintenance/ai/start")
        self.assertEqual(500, response.status_code)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertIn("Container not found.", data["errors"])


class DiagnosticsPageTests(unittest.TestCase):
    """Tests for the /diagnostics page and /api/diagnostics/ai-progress endpoint."""

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    def test_diagnostics_page_renders(self) -> None:
        client = self._app().test_client()
        response = client.get("/diagnostics")
        self.assertEqual(200, response.status_code)
        self.assertIn(b"Diagnostics", response.data)

    @mock.patch("web_app._check_ai_status", return_value="ready")
    def test_ai_progress_endpoint_returns_ready_status(self, _mock: mock.Mock) -> None:
        client = self._app().test_client()
        response = client.get("/api/diagnostics/ai-progress")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertEqual("ready", data["ai_status"])
        self.assertIn("extraction_provider", data)
        self.assertIn("ai_model", data)
        self.assertIn("ai_endpoint", data)
        self.assertIsNone(data["pull_progress"])

    @mock.patch("web_app._get_ai_pull_progress")
    @mock.patch("web_app._check_ai_status", return_value="loading")
    def test_ai_progress_endpoint_includes_pull_progress_when_loading(
        self, _status_mock: mock.Mock, progress_mock: mock.Mock
    ) -> None:
        progress_mock.return_value = {
            "available": True,
            "percent": 42.0,
            "status_message": "pulling layer",
            "current_bytes": 420000000,
            "total_bytes": 1000000000,
            "layers": 3,
        }
        client = self._app().test_client()
        response = client.get("/api/diagnostics/ai-progress")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertEqual("loading", data["ai_status"])
        pp = data["pull_progress"]
        self.assertIsNotNone(pp)
        self.assertEqual(42.0, pp["percent"])
        self.assertEqual(3, pp["layers"])
        progress_mock.assert_called_once()

    @mock.patch("web_app._check_ai_status", return_value="unavailable")
    def test_ai_progress_endpoint_no_pull_progress_when_unavailable(
        self, _mock: mock.Mock
    ) -> None:
        client = self._app().test_client()
        response = client.get("/api/diagnostics/ai-progress")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertEqual("unavailable", data["ai_status"])
        self.assertIsNone(data["pull_progress"])


class PullProgressParsingTests(unittest.TestCase):
    """Tests for _parse_pull_progress_from_logs."""

    def setUp(self) -> None:
        from web_app import _parse_pull_progress_from_logs
        self._parse = _parse_pull_progress_from_logs

    def test_parses_layer_progress_from_json_lines(self) -> None:
        lines = [
            '{"status":"pulling manifest"}',
            '{"status":"pulling abc123","digest":"sha256:abc","total":1000,"completed":400}',
            '{"status":"pulling abc123","digest":"sha256:abc","total":1000,"completed":800}',
        ]
        result = self._parse(lines)
        self.assertEqual(80.0, result["percent"])
        self.assertEqual(800, result["current_bytes"])
        self.assertEqual(1000, result["total_bytes"])
        self.assertEqual(1, result["layers"])
        self.assertEqual("pulling abc123", result["status_message"])

    def test_aggregates_multiple_layers(self) -> None:
        lines = [
            '{"status":"pulling layer1","digest":"sha256:aaa","total":1000,"completed":1000}',
            '{"status":"pulling layer2","digest":"sha256:bbb","total":1000,"completed":500}',
        ]
        result = self._parse(lines)
        self.assertEqual(75.0, result["percent"])
        self.assertEqual(1500, result["current_bytes"])
        self.assertEqual(2000, result["total_bytes"])
        self.assertEqual(2, result["layers"])

    def test_handles_timestamped_docker_log_lines(self) -> None:
        lines = [
            '2024-01-01T00:00:00.000000000Z {"status":"pulling manifest"}',
            '2024-01-01T00:00:01.000000000Z {"status":"pulling abc","digest":"sha256:xyz","total":500,"completed":250}',
        ]
        result = self._parse(lines)
        self.assertEqual(50.0, result["percent"])

    def test_returns_none_percent_when_no_total(self) -> None:
        lines = ['{"status":"pulling manifest"}']
        result = self._parse(lines)
        self.assertIsNone(result["percent"])
        self.assertEqual(0, result["total_bytes"])
        self.assertEqual(0, result["layers"])

    def test_handles_empty_log_lines(self) -> None:
        result = self._parse([])
        self.assertIsNone(result["percent"])
        self.assertEqual(0, result["layers"])
        self.assertEqual("", result["status_message"])

    def test_uses_last_status_from_plain_text_lines(self) -> None:
        lines = [
            'some plain text line',
            'verifying sha256 digest',
        ]
        result = self._parse(lines)
        self.assertEqual("verifying sha256 digest", result["status_message"])

    def test_handles_malformed_json_gracefully(self) -> None:
        lines = [
            '{"status": "ok"}',
            '{not valid json',
            '{"status":"pulling x","digest":"sha256:d","total":100,"completed":50}',
        ]
        result = self._parse(lines)
        self.assertEqual(50.0, result["percent"])


class AnnotationQueueRouteTests(unittest.TestCase):
    """Tests for /annotate and related new routes."""

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    @mock.patch("web_app.get_annotation_queue", return_value=[])
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_annotate_route_returns_200(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _queue_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.get("/annotate")
        self.assertEqual(200, response.status_code)
        self.assertIn(b"Annotation Queue", response.data)

    @mock.patch("web_app.get_annotation_queue", return_value=[])
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_annotate_route_respects_pagination(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        queue_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        client.get("/annotate?limit=10&offset=20")
        queue_mock.assert_called_once()
        _args, kwargs = queue_mock.call_args
        self.assertEqual(10, kwargs["limit"])
        self.assertEqual(20, kwargs["offset"])


class TrainingDashboardRouteTests(unittest.TestCase):
    """Tests for /training and /api/training/* routes."""

    _EMPTY_STATS = {
        "total_gt_corrections": 0,
        "by_field": [],
        "by_state": [],
        "coverage": {"total_records": 0, "ge1": 0, "ge5": 0, "ge10": 0},
    }

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    @mock.patch("web_app.list_training_runs", return_value=[])
    @mock.patch("web_app.get_training_stats", return_value=_EMPTY_STATS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_training_page_returns_200(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _stats_mock: mock.Mock,
        _runs_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.get("/training")
        self.assertEqual(200, response.status_code)
        self.assertIn(b"Training Dashboard", response.data)

    @mock.patch("web_app.get_training_stats", return_value=_EMPTY_STATS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_api_training_stats_returns_json(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _stats_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.get("/api/training/stats")
        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertIn("total_gt_corrections", data)
        self.assertIn("coverage", data)

    @mock.patch("web_app.insert_training_run", return_value=1)
    @mock.patch("web_app.list_ground_truth_corrections", return_value=[])
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_api_training_export_returns_zip(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
        _run_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.post("/api/training/export", data={"notes": "test export"})
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/zip", response.content_type)


class ManualEntryRouteTests(unittest.TestCase):
    """Tests for manual_entry=1 upload mode."""

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    @mock.patch("web_app.insert_title_record")
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_manual_entry_redirects_to_review(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        insert_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        insert_mock.return_value = {"id": 99, "is_validated": False, "validation_errors": ["VIN"]}
        client = self._app().test_client()

        response = client.post(
            "/",
            data={
                "file": (BytesIO(b"fake-image"), "title.png"),
                "state": "NM",
                "manual_entry": "1",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(302, response.status_code)
        self.assertIn("/review/99", response.headers["Location"])
        insert_mock.assert_called_once()

    @mock.patch("web_app._run_extraction")
    @mock.patch("web_app.create_connection_from_env")
    def test_normal_upload_skips_manual_entry(
        self,
        conn_mock: mock.Mock,
        extraction_mock: mock.Mock,
    ) -> None:
        extraction_mock.return_value = None
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()

        response = client.post(
            "/",
            data={"file": (BytesIO(b"fake"), "title.png")},
            content_type="multipart/form-data",
        )

        extraction_mock.assert_called_once()
        self.assertEqual(200, response.status_code)


class ReviewRecordBulkApproveTests(unittest.TestCase):
    """Tests for bulk_approve flag in /review/<id> POST."""

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    @mock.patch("web_app.update_title_record_fields")
    @mock.patch("web_app.insert_correction")
    @mock.patch("web_app.get_record_by_id")
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_bulk_approve_marks_corrections_as_ground_truth(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        get_record_mock: mock.Mock,
        insert_mock: mock.Mock,
        update_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        get_record_mock.return_value = {
            "id": 1, "state": "NM", "title_number": "T123", "vin": "1HGCM82633A004352",
            "vehicle_year": 2003, "source_file_path": None,
            **{f: None for f in ("make", "model", "body_style", "color", "odometer",
                                 "owner_name", "owner_address", "purchase_price",
                                 "sale_date", "issue_date", "state_layout_version",
                                 "ocr_text", "is_validated", "validation_errors",
                                 "created_at")},
        }
        insert_mock.return_value = 1
        client = self._app().test_client()

        response = client.post(
            "/review/1",
            data={"field_vin": "1HGCM82633A004352", "bulk_approve": "1"},
        )

        self.assertEqual(200, response.status_code)
        insert_mock.assert_called_once()
        _call_kwargs = insert_mock.call_args[1]
        self.assertTrue(_call_kwargs.get("is_ground_truth"))


class ApplyToLLMTests(unittest.TestCase):
    """Tests for _build_llm_modelfile() and POST /api/training/apply-to-llm."""

    # ── _build_llm_modelfile unit tests ───────────────────────────────────────

    def _make_correction(self, record_id: int, field: str, value: str) -> dict:
        return {
            "record_id": record_id,
            "field_name": field,
            "corrected_value": value,
            "source_file_path": None,
        }

    def test_modelfile_starts_with_from_base_model(self) -> None:
        result = _build_llm_modelfile([], "moondream")
        self.assertTrue(result.startswith("FROM moondream"))

    def test_modelfile_contains_system_block(self) -> None:
        result = _build_llm_modelfile([], "moondream")
        self.assertIn('SYSTEM """', result)
        self.assertIn("vehicle title OCR assistant", result)
        self.assertIn("title_number", result)

    def test_modelfile_with_no_corrections_has_no_message_pairs(self) -> None:
        result = _build_llm_modelfile([], "moondream")
        self.assertNotIn("MESSAGE", result)

    def test_modelfile_adds_message_pairs_for_each_record(self) -> None:
        corrections = [
            self._make_correction(1, "state", "NM"),
            self._make_correction(1, "vin", "1HGCM82633A004352"),
            self._make_correction(2, "state", "TX"),
        ]
        result = _build_llm_modelfile(corrections, "moondream")
        # Two distinct record_ids → two MESSAGE user/assistant pairs.
        self.assertEqual(2, result.count("MESSAGE user"))
        self.assertEqual(2, result.count("MESSAGE assistant"))

    def test_modelfile_assistant_message_contains_corrected_values(self) -> None:
        corrections = [self._make_correction(1, "state", "NM")]
        result = _build_llm_modelfile(corrections, "moondream")
        self.assertIn('"NM"', result)

    def test_modelfile_caps_examples_at_max(self) -> None:
        from web_app import _LLM_MAX_EXAMPLES
        corrections = [
            self._make_correction(i, "state", "NM") for i in range(_LLM_MAX_EXAMPLES + 10)
        ]
        result = _build_llm_modelfile(corrections, "moondream")
        self.assertEqual(_LLM_MAX_EXAMPLES, result.count("MESSAGE user"))

    def test_modelfile_sanitises_triple_quotes_in_values(self) -> None:
        corrections = [self._make_correction(1, "owner_name", 'Bad"""Value')]
        result = _build_llm_modelfile(corrections, "moondream")
        # Triple-quotes in the value must be stripped; the Modelfile must
        # still contain exactly one MESSAGE block.
        self.assertEqual(1, result.count("MESSAGE assistant"))
        self.assertNotIn('Bad"""Value', result)
        # Value with triple-quotes removed ("BadValue") must appear.
        self.assertIn("BadValue", result)

    def test_modelfile_includes_all_14_fields_in_assistant_json(self) -> None:
        corrections = [self._make_correction(1, "state", "NM")]
        result = _build_llm_modelfile(corrections, "moondream")
        for field in (
            "state", "title_number", "vin", "vehicle_year", "make", "model",
            "body_style", "color", "odometer", "owner_name", "owner_address",
            "purchase_price", "sale_date", "issue_date",
        ):
            self.assertIn(f'"{field}"', result)

    # ── /api/training/apply-to-llm endpoint tests ────────────────────────────

    def _app(self):
        app = create_app()
        app.testing = True
        return app

    _SAMPLE_CORRECTIONS = [
        {
            "record_id": 1,
            "field_name": "state",
            "corrected_value": "NM",
            "source_file_path": None,
        }
    ]

    @mock.patch("web_app.list_ground_truth_corrections", return_value=[])
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_returns_400_when_no_corrections(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.post("/api/training/apply-to-llm")
        self.assertEqual(400, response.status_code)
        data = json.loads(response.data)
        self.assertIn("error", data)

    @mock.patch("web_app.insert_training_run", return_value=7)
    @mock.patch("web_app.requests.post")
    @mock.patch("web_app.list_ground_truth_corrections", return_value=_SAMPLE_CORRECTIONS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_returns_model_name_on_success(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
        post_mock: mock.Mock,
        _run_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        ollama_resp = mock.Mock()
        ollama_resp.raise_for_status = mock.Mock()
        post_mock.return_value = ollama_resp

        client = self._app().test_client()
        response = client.post("/api/training/apply-to-llm")

        self.assertEqual(200, response.status_code)
        data = json.loads(response.data)
        self.assertEqual("titles-custom", data["model"])
        self.assertIn("message", data)
        self.assertIn("run_id", data)
        self.assertEqual(7, data["run_id"])

    @mock.patch("web_app.insert_training_run", return_value=1)
    @mock.patch("web_app.requests.post")
    @mock.patch("web_app.list_ground_truth_corrections", return_value=_SAMPLE_CORRECTIONS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_posts_modelfile_to_ollama(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
        post_mock: mock.Mock,
        _run_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        ollama_resp = mock.Mock()
        ollama_resp.raise_for_status = mock.Mock()
        post_mock.return_value = ollama_resp

        self._app().test_client().post("/api/training/apply-to-llm")

        post_mock.assert_called_once()
        call_kwargs = post_mock.call_args
        payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
        self.assertEqual("titles-custom", payload["name"])
        self.assertIn("modelfile", payload)
        self.assertFalse(payload.get("stream", True))

    @mock.patch(
        "web_app.requests.post",
        side_effect=__import__("requests").exceptions.ConnectionError("refused"),
    )
    @mock.patch("web_app.list_ground_truth_corrections", return_value=_SAMPLE_CORRECTIONS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_returns_503_when_ollama_unavailable(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
        _post_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.post("/api/training/apply-to-llm")
        self.assertEqual(503, response.status_code)
        data = json.loads(response.data)
        self.assertIn("error", data)

    @mock.patch(
        "web_app.requests.post",
        side_effect=__import__("requests").exceptions.Timeout("timed out"),
    )
    @mock.patch("web_app.list_ground_truth_corrections", return_value=_SAMPLE_CORRECTIONS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_returns_504_when_ollama_times_out(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
        _post_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        client = self._app().test_client()
        response = client.post("/api/training/apply-to-llm")
        self.assertEqual(504, response.status_code)
        data = json.loads(response.data)
        self.assertIn("error", data)

    @mock.patch("web_app.insert_training_run", return_value=1)
    @mock.patch("web_app.requests.post")
    @mock.patch("web_app.list_ground_truth_corrections", return_value=_SAMPLE_CORRECTIONS)
    @mock.patch("web_app.initialize_database")
    @mock.patch("web_app.create_connection_from_env")
    def test_records_training_run_on_success(
        self,
        conn_mock: mock.Mock,
        _init_mock: mock.Mock,
        _corr_mock: mock.Mock,
        post_mock: mock.Mock,
        run_mock: mock.Mock,
    ) -> None:
        conn_mock.return_value = mock.Mock()
        ollama_resp = mock.Mock()
        ollama_resp.raise_for_status = mock.Mock()
        post_mock.return_value = ollama_resp

        self._app().test_client().post("/api/training/apply-to-llm")

        run_mock.assert_called_once()
        _args, kwargs = run_mock.call_args
        self.assertIn("LLM prompt-tuning", kwargs.get("notes", _args[2] if len(_args) > 2 else ""))


if __name__ == "__main__":
    unittest.main()

