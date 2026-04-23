import json
import unittest
from io import BytesIO
from unittest import mock

from web_app import (
    _apply_word_confidence,
    _check_ai_status,
    _docker_client,
    _extract_text_from_image,
    _extract_text_from_pdf,
    _get_service_containers,
    _get_service_logs,
    _get_word_confidences,
    _run_extraction,
    _start_ai_services,
    _tesseract_fields_and_confidence,
    allowed_file,
    create_app,
    extract_text_from_upload,
    get_port_from_environment,
    parse_extracted_fields,
)

_GOOD_VIN = "1HGCM82633A004352"
_GOOD_OCR = f"VIN {_GOOD_VIN} YEAR 2003 TITLE ABC1234"
_GOOD_EXTRACTION = {
    "fields": {
        "state": None,
        "title_number": "ABC1234",
        "vin": _GOOD_VIN,
        "vehicle_year": 2003,
    },
    "raw_text": _GOOD_OCR,
    "source": "tesseract",
    "confidence": {"state": 0.0, "title_number": 0.7, "vin": 0.7, "vehicle_year": 0.7},
    "low_confidence": ["state"],
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
        from PIL import Image as _Image

        fake_image = _Image.new("RGB", (100, 80), color=(200, 200, 200))
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
        image_open_mock.return_value = object()

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
        self.assertEqual([], result["low_confidence"])

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


if __name__ == "__main__":
    unittest.main()
