import unittest

from web_app import allowed_file, parse_extracted_fields


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


if __name__ == "__main__":
    unittest.main()
