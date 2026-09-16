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


def synthetic_line(text: str, x: int, y: int, width: int = 100, height: int = 20) -> dict:
    return {"text": text, "score": 0.99, "box": [x, y, x + width, y + height]}


class AdapterTests(unittest.TestCase):
    def test_bare_integer_fare_in_a_dated_row_is_not_dropped(self) -> None:
        lines = fixture_lines("multi_ride")
        lines[3]["text"] = "15"
        result = extract_fields(lines, "ride_payment", business_context={"currency": "RMB"})
        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual(result["candidate_totals"][0]["amount"], "42.75")
        self.assertEqual(result["candidate_totals"][0]["component_count"], 2)

    def test_times_dates_and_identifiers_cannot_fill_a_missing_fare(self) -> None:
        for text in ("12:30", "20260802", "ID 1234", "2026", "1234567890123456789"):
            with self.subTest(text=text):
                lines = fixture_lines("multi_ride")
                lines[3]["text"] = text
                result = extract_fields(lines, "ride_payment", business_context={"currency": "CNY"})
                self.assertEqual(result["profile_check"], "review")
                self.assertEqual(len(result["transactions"]), 1)
                self.assertTrue(any("Incomplete transaction coverage" in warning for warning in result["warnings"]))

    def test_ride_cards_pair_fares_above_dates_and_parse_glued_times(self) -> None:
        lines = [synthetic_line("呼叫返程", 10, 0)]
        for offset, amount, timestamp in ((0, "15", "2026-10-0514:25"), (180, "¥16.20", "2026-10-0509:12")):
            lines.extend([
                synthetic_line(amount, 350, 50 + offset, 70),
                synthetic_line("起点：示例起点", 10, 60 + offset, 200),
                synthetic_line("终点：示例终点", 10, 90 + offset, 200),
                synthetic_line("下单时间：" + timestamp, 10, 120 + offset, 200),
            ])
        result = extract_fields(lines, "ride_payment", business_context={"currency": "CNY"})
        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual([row["date"]["value"] for row in result["transactions"]], ["2026-10-05"] * 2)
        self.assertEqual(result["candidate_totals"][0]["amount"], "31.20")

    def test_unassociated_second_fare_requires_review(self) -> None:
        lines = fixture_lines("multi_ride")
        lines[4]["text"] = "Unrecognized date"
        result = extract_fields(lines, "ride_payment")
        self.assertEqual(result["profile_check"], "review")
        self.assertTrue(any("Unassociated" in warning for warning in result["warnings"]))

    def test_explicit_transit_context_supports_headerless_layout_but_not_missing_sign(self) -> None:
        lines = [synthetic_line("2026-10-0614:25", 10, 55, 180), synthetic_line("- 12", 310, 40, 70, 28)]
        self.assertEqual(extract_fields(lines, "transit_payment")["profile_check"], "unsupported")
        context = {"provider": "octopus", "currency": "HKD"}
        result = extract_fields(lines, "transit_payment", business_context=context)
        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual(result["candidate_totals"][0]["amount"], "-12.00")
        lines[1]["text"] = "12"
        result = extract_fields(lines, "transit_payment", business_context=context)
        self.assertEqual(result["profile_check"], "review")
        self.assertEqual(result["transactions"][0]["amount"]["value"], "12.00")

    def test_alipay_paid_amount_has_priority_over_order_total(self) -> None:
        lines = fixture_lines("alipay_payment")
        lines[2]["text"] = "订单金额"
        lines.extend([synthetic_line("实付金额", 500, 70), synthetic_line("58.20", 500, 105)])
        result = extract_fields(lines, "alipay", business_context={"currency": "CNY"})
        self.assertEqual(result["fields"]["amount"]["value"], "58.20")
        lines[-1]["text"] = "Unreadable paid amount"
        result = extract_fields(lines, "alipay", business_context={"currency": "CNY"})
        self.assertEqual(result["profile_check"], "review")
        self.assertNotIn("amount", result["fields"])

    def test_labeled_money_preserves_negative_sign_and_integer_year_values(self) -> None:
        for raw, expected in (("-2026", "-2026.00"), ("2026", "2026.00")):
            lines = fixture_lines("alipay_payment")
            lines[2]["text"] = "实付金额：" + raw
            result = extract_fields(lines, "alipay", business_context={"currency": "CNY"})
            self.assertEqual(result["fields"]["amount"]["value"], expected)

    def test_xianyu_chinese_price_and_wrapped_trade_number(self) -> None:
        lines = [
            synthetic_line("闲鱼", 10, 0),
            synthetic_line("成交价(已到达卖家账户)", 10, 50, 210),
            synthetic_line("¥42.00", 350, 50),
            synthetic_line("订单编号", 10, 100),
            synthetic_line("SYN-ORDER-002", 250, 100, 200),
            synthetic_line("支付宝交易号", 10, 150, 140),
            synthetic_line("209901010000000000000", 200, 150, 210),
            synthetic_line("0000000", 330, 180, 80),
        ]
        result = extract_fields(lines, "xianyu", expected_fields=["amount", "order_id", "transaction_id"])
        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual(result["fields"]["amount"]["value"], "42.00")
        self.assertEqual(result["fields"]["transaction_id"]["value"], "2099010100000000000000000000")
        self.assertEqual(len(result["fields"]["transaction_id"]["source_boxes"]), 2)

    def test_merchant_order_date_and_payment_date_remain_distinct(self) -> None:
        lines = fixture_lines("taobao_order")
        lines.extend([
            synthetic_line("创建时间", 10, 260),
            synthetic_line("2026-10-0823:55:00", 200, 260, 220),
            synthetic_line("付款时间", 10, 300),
            synthetic_line("2026-10-09 00:05:00", 200, 300, 220),
        ])
        result = extract_fields(lines, "taobao")
        self.assertEqual(result["fields"]["order_date"]["value"], "2026-10-08")
        self.assertEqual(result["fields"]["paid_date"]["value"], "2026-10-09")
        lines[-2:] = []
        result = extract_fields(lines, "taobao")
        self.assertNotIn("paid_date", result["fields"])

    def test_xianyu_order_time_does_not_require_or_invent_payment_method(self) -> None:
        lines = fixture_lines("xianyu_order")[:5]
        lines.extend([
            synthetic_line("下单时间", 10, 150),
            synthetic_line("2026-10-1016:53:48", 200, 150, 220),
        ])
        result = extract_fields(lines, "xianyu")
        self.assertEqual(result["profile_check"], "pass")
        self.assertEqual(result["fields"]["order_date"]["value"], "2026-10-10")
        self.assertNotIn("paid_date", result["fields"])
        self.assertNotIn("payment_method", result["fields"])

    def test_approval_table_preserves_separate_ranges_and_detects_missing_rows(self) -> None:
        lines = [synthetic_line("Destination", 10, 10), synthetic_line('["Start Time","End Time"]', 220, 10, 220), synthetic_line("审批状态", 700, 10)]
        for y, start, end in ((70, "2026-10-01", "2026-10-03"), (150, "2026-10-06", "2026-10-07")):
            lines.extend([synthetic_line("示例城市", 10, y), synthetic_line(f'["{start} 上午","{end}', 220, y - 8, 300), synthetic_line("审批通过", 700, y)])
        result = extract_fields(lines, "travel_approval", expected_fields=["approvals"])
        self.assertEqual(result["profile_check"], "pass")
        approvals = result["fields"]["approvals"]["value"]
        self.assertEqual(len(approvals), 2)
        self.assertEqual(approvals[1]["start_date"]["value"], "2026-10-06")
        self.assertNotIn("start_date", result["fields"])
        lines[7]["text"] = "Unrecognized interval"
        result = extract_fields(lines, "travel_approval", expected_fields=["approvals"])
        self.assertEqual(result["profile_check"], "review")
        self.assertTrue(any("Incomplete approval coverage" in warning for warning in result["warnings"]))

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
