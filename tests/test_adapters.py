from __future__ import annotations

import json
import unittest
from pathlib import Path

from hkclr_rapidocr_eval.adapters import PROFILE_ADAPTERS, extract_fields

FIXTURES = Path(__file__).parent / "fixtures" / "ocr"


def fixture_lines(name: str) -> list[dict[str, object]]:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    if payload.get("synthetic") is not True:
        raise AssertionError("OCR fixtures must be explicitly synthetic")
    return payload["lines"]


class AdapterTests(unittest.TestCase):
    def test_registry_contains_every_business_profile(self) -> None:
        self.assertEqual(
            set(PROFILE_ADAPTERS),
            {
                "taobao_order_detail",
                "xianyu_order_detail",
                "alipay_payment_detail",
                "vendor_receipt",
                "travel_approval",
                "ride_payment",
                "transit_payment",
            },
        )

    def test_travel_approval_uses_boxes_to_associate_values(self) -> None:
        result = extract_fields(fixture_lines("travel_approval"), "travel_approval")

        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual(result["fields"]["destination"]["value"], "Shenzhen")
        self.assertEqual(result["fields"]["start_date"]["value"], "2026-10-05")
        self.assertEqual(result["fields"]["end_date"]["value"], "2026-10-07")
        self.assertEqual(result["fields"]["approval_status"]["value"], "Approved")
        self.assertEqual(result["fields"]["destination"]["source_box"][0], [180, 50])

    def test_multi_ride_rows_have_a_deterministic_total(self) -> None:
        result = extract_fields(fixture_lines("multi_ride"), "ride_payment")

        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual(len(result["transactions"]), 2)
        self.assertEqual(result["transactions"][0]["date"]["value"], "2026-08-01")
        self.assertEqual(result["transactions"][1]["amount"]["value"], "27.75")
        self.assertEqual(
            result["candidate_totals"],
            [
                {
                    "kind": "transaction_sum",
                    "amount": "46.25",
                    "currency": "CNY",
                    "component_count": 2,
                    "source_boxes": [
                        [[300, 50], [390, 50], [390, 70], [300, 70]],
                        [[300, 85], [390, 85], [390, 105], [300, 105]],
                    ],
                }
            ],
        )

    def test_transit_rows_preserve_signed_debits_without_amount_labels(self) -> None:
        result = extract_fields(fixture_lines("transit_payment"), "transit_payment")

        self.assertEqual(
            [row["amount"]["value"] for row in result["transactions"]],
            ["-12.30", "-7.70"],
        )
        self.assertEqual(
            [row["amount"]["currency"] for row in result["transactions"]],
            ["HKD", "HKD"],
        )
        self.assertEqual(result["candidate_totals"][0]["amount"], "-20.00")

    def test_vendor_receipt_extracts_required_fields(self) -> None:
        result = extract_fields(fixture_lines("vendor_receipt"), "vendor_receipt")

        values = {name: field["value"] for name, field in result["fields"].items()}
        self.assertEqual(
            values,
            {
                "receipt_id": "SYN-RCPT-001",
                "paid_date": "2026-08-05",
                "amount": "128.40",
                "currency": "HKD",
                "payment_method": "Corporate Card",
            },
        )
        self.assertEqual(result["candidate_totals"][0]["currency"], "HKD")

    def test_existing_marketplace_and_payment_profiles_remain_covered(self) -> None:
        cases = (
            ("taobao_order", "taobao", "123.45"),
            ("alipay_payment", "alipay", "66.60"),
            ("xianyu_order", "xianyu_order_detail", "88.80"),
        )
        for fixture, profile, amount in cases:
            with self.subTest(profile=profile):
                result = extract_fields(fixture_lines(fixture), profile)
                self.assertEqual(result["profile_check"], "pass")
                self.assertEqual(result["fields"]["amount"]["value"], amount)
                self.assertEqual(result["fields"]["amount"]["currency"], "CNY")

    def test_unsupported_layout_cannot_pass(self) -> None:
        result = extract_fields(fixture_lines("unsupported"), "vendor_receipt")

        self.assertEqual(result["support_status"], "unsupported")
        self.assertEqual(result["profile_check"], "unsupported")
        self.assertEqual(result["fields"], {})


if __name__ == "__main__":
    unittest.main()
