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
            {"text": "支付宝交易号 2099010100000000000000000000", "score": 0.96},
        ]

        fields = extract_fields(lines, "taobao")

        self.assertEqual(fields["profile_check"], "pass")
        self.assertEqual(fields["fields"]["amount"]["value"], "123.45")
        self.assertEqual(fields["fields"]["amount"]["currency"], "CNY")
        self.assertEqual(
            fields["fields"]["transaction_id"]["value"], "2099010100000000000000000000"
        )
        self.assertNotIn("amount_candidates", fields)

    def test_unknown_layout_is_unsupported(self) -> None:
        fields = extract_fields([{"text": "交易成功", "score": 0.99}], "alipay")
        self.assertEqual(fields["profile_check"], "unsupported")
        self.assertEqual(fields["support_status"], "unsupported")

    def test_profile_inference(self) -> None:
        self.assertEqual(
            profile_for(Path("01_alipay_record.png"), "auto"), "alipay_payment_detail"
        )
        self.assertEqual(
            profile_for(Path("01_淘宝订单.png"), "auto"), "taobao_order_detail"
        )
        self.assertEqual(
            profile_for(Path("物品/hqchip/01_order_panel.png"), "auto"), "generic"
        )
        self.assertEqual(profile_for(Path("receipt.png"), "auto"), "vendor_receipt")

    def test_discovers_supported_images_only(self) -> None:
        root = Path(__file__).parent / "fixtures"

        found = discover_images(root)

        self.assertEqual({path.name for path in found}, {"a.PGM", "b.pgm"})


if __name__ == "__main__":
    unittest.main()
