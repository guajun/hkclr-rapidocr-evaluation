from __future__ import annotations

import unittest
from pathlib import Path

from hkclr_rapidocr_eval.core import discover_images, extract_fields, profile_for


class CoreTests(unittest.TestCase):
    def test_extracts_taobao_trade_number_and_amount(self) -> None:
        lines = [
            {"text": "交易成功", "score": 0.99},
            {"text": "实付款 ￥123.45", "score": 0.98},
            {"text": "支付方式 支付宝支付", "score": 0.97},
            {"text": "支付宝交易号 2026071712345678901234567890", "score": 0.96},
        ]

        fields = extract_fields(lines, "taobao")

        self.assertEqual(fields["profile_check"], "pass")
        self.assertEqual(fields["amount_candidates"], ["123.45"])
        self.assertEqual(fields["alipay_trade_no_candidates"], ["2026071712345678901234567890"])

    def test_missing_keyword_requires_review(self) -> None:
        fields = extract_fields([{"text": "交易成功", "score": 0.99}], "alipay")
        self.assertEqual(fields["profile_check"], "review")
        self.assertIn("流水号", fields["required_keywords_missing"])

    def test_profile_inference(self) -> None:
        self.assertEqual(profile_for(Path("01_payment_record.png"), "auto"), "alipay")
        self.assertEqual(profile_for(Path("01_淘宝订单.png"), "auto"), "taobao")
        self.assertEqual(profile_for(Path("receipt.png"), "auto"), "generic")

    def test_discovers_supported_images_only(self) -> None:
        root = Path(__file__).parent / "fixtures"

        found = discover_images(root)

        self.assertEqual({path.name for path in found}, {"a.PGM", "b.pgm"})


if __name__ == "__main__":
    unittest.main()
