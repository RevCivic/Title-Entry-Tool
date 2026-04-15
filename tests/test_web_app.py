import unittest
from unittest import mock

from web_app import (
    _extract_text_from_image,
    _extract_text_from_pdf,
    allowed_file,
    extract_text_from_upload,
    parse_extracted_fields,
)


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

    @mock.patch("web_app._extract_text_from_pdf", return_value="pdf text")
    @mock.patch("web_app._extract_text_from_image", return_value="image text")
    def test_extract_text_from_upload_routes_by_extension(
        self,
        image_mock: mock.Mock,
        pdf_mock: mock.Mock,
    ) -> None:
        self.assertEqual("pdf text", extract_text_from_upload("sample.PDF", b"x"))
        self.assertEqual("image text", extract_text_from_upload("sample.png", b"x"))
        pdf_mock.assert_called_once_with(b"x")
        image_mock.assert_called_once_with(b"x")

    @mock.patch("web_app.pytesseract.image_to_string", return_value="ocr image text")
    @mock.patch("web_app.Image.open")
    def test_extract_text_from_image_uses_tesseract(
        self,
        image_open_mock: mock.Mock,
        image_to_string_mock: mock.Mock,
    ) -> None:
        fake_image = object()
        image_open_mock.return_value = fake_image
        result = _extract_text_from_image(b"img-bytes")
        self.assertEqual("ocr image text", result)
        image_to_string_mock.assert_called_once_with(fake_image)

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


if __name__ == "__main__":
    unittest.main()
