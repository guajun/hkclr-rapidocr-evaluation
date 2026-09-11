"""Business-aware OCR profile adapters and spatial extraction helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .schemas import ADAPTER_REGISTRY_VERSION, ADAPTER_SCHEMA

TRADE_NO_RE = re.compile(r"(?<!\d)20\d{26}(?!\d)")
DATE_RE = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:[-/.年])\s*(\d{1,2})\s*(?:[-/.月])\s*(\d{1,2})\s*日?(?!\d)"
)
AMOUNT_RE = re.compile(
    r"(?<![\dA-Za-z])"
    r"(?P<sign_before>[+-])?\s*"
    r"(?P<currency>HK\$|US\$|CN¥|RMB|CNY|HKD|USD|MOP|JPY|NTD|¥|￥|\$)?\s*"
    r"(?P<sign_after>[+-])?\s*"
    r"(?P<number>\d{1,8}(?:,\d{3})*(?:\.\d{1,2})?)"
    r"(?!\d)",
    flags=re.IGNORECASE,
)
AMOUNT_MARKERS = (
    "金额",
    "金額",
    "实付",
    "實付",
    "付款",
    "合计",
    "合計",
    "total",
    "amount",
)
TOTAL_MARKERS = (
    "合计",
    "合計",
    "总计",
    "總計",
    "total",
    "实付",
    "實付",
    "订单金额",
    "訂單金額",
)

Field = dict[str, Any]
Lines = list[dict[str, Any]]
ExtractorResult = tuple[
    dict[str, Field],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]
Extractor = Callable[[Lines, dict[str, Any]], ExtractorResult]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("：", ":").casefold()


def _box(line: dict[str, Any]) -> list[Any] | None:
    box = line.get("box")
    return box if isinstance(box, list) and box else None


def _bounds(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = _box(line)
    if box is None:
        return None
    try:
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            left, top, right, bottom = box
            return float(left), float(top), float(right), float(bottom)
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return min(xs), min(ys), max(xs), max(ys)
    except (IndexError, TypeError, ValueError):
        return None


def _confidence(line: dict[str, Any]) -> float:
    try:
        return round(float(line.get("score", 0.0)), 4)
    except (TypeError, ValueError):
        return 0.0


def _field(value: Any, line: dict[str, Any], *, raw: str | None = None) -> Field:
    result: Field = {
        "value": value,
        "confidence": _confidence(line),
        "source_box": _box(line),
    }
    if raw is not None:
        result["raw"] = raw
    return result


def _normalize_date(match: re.Match[str]) -> str:
    year, month, day = (int(value) for value in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def _date_from_text(text: str, line: dict[str, Any]) -> Field | None:
    match = DATE_RE.search(text)
    return _field(_normalize_date(match), line, raw=match.group(0)) if match else None


def _currency_code(token: str | None) -> str | None:
    if token is None:
        return None
    normalized = token.upper()
    return {
        "RMB": "CNY",
        "CN¥": "CNY",
        "¥": "CNY",
        "￥": "CNY",
        "HK$": "HKD",
        "US$": "USD",
        "$": None,
    }.get(normalized, normalized)


def _money_from_match(
    match: re.Match[str], line: dict[str, Any], business_context: dict[str, Any]
) -> Field | None:
    try:
        amount = Decimal(match.group("number").replace(",", ""))
    except InvalidOperation:
        return None
    sign = match.group("sign_before") or match.group("sign_after")
    if sign == "-":
        amount = -amount
    currency_token = match.group("currency")
    currency = _currency_code(currency_token)
    currency_source = "ocr" if currency is not None else None
    if currency is None and isinstance(business_context.get("currency"), str):
        currency = business_context["currency"].upper()
        currency_source = "business_context"
    result = _field(f"{amount:.2f}", line, raw=match.group(0).strip())
    result.update(
        currency=currency,
        currency_token=currency_token,
        currency_source=currency_source,
        explicit_sign=sign,
    )
    return result


def _amounts_in_line(
    line: dict[str, Any], business_context: dict[str, Any], *, allow_plain: bool = False
) -> list[Field]:
    text = str(line.get("text", ""))
    date_spans = [match.span() for match in DATE_RE.finditer(text)]
    has_marker = any(marker in text.casefold() for marker in AMOUNT_MARKERS)
    results: list[Field] = []
    for match in AMOUNT_RE.finditer(text):
        if any(
            match.start() < end and match.end() > start for start, end in date_spans
        ):
            continue
        number = match.group("number")
        if not (
            allow_plain
            or has_marker
            or match.group("currency")
            or match.group("sign_before")
            or match.group("sign_after")
            or "." in number
        ):
            continue
        amount = _money_from_match(match, line, business_context)
        if amount is not None:
            results.append(amount)
    return results


def _inline_suffix(text: str, label: str) -> str | None:
    start = text.casefold().find(label.casefold())
    if start < 0:
        return None
    suffix = text[start + len(label) :].lstrip(" \t:：-—")
    return suffix or None


def _nearby_lines(lines: Lines, label_index: int) -> list[dict[str, Any]]:
    label = lines[label_index]
    label_bounds = _bounds(label)
    if label_bounds is None:
        return lines[label_index + 1 : label_index + 2]

    left, top, right, bottom = label_bounds
    height = max(bottom - top, 1.0)
    center_y = (top + bottom) / 2
    ranked: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for index, candidate in enumerate(lines):
        if index == label_index:
            continue
        bounds = _bounds(candidate)
        if bounds is None:
            continue
        candidate_left, candidate_top, candidate_right, candidate_bottom = bounds
        candidate_height = max(candidate_bottom - candidate_top, 1.0)
        candidate_center_y = (candidate_top + candidate_bottom) / 2
        same_row = (
            abs(candidate_center_y - center_y) <= max(height, candidate_height) * 0.75
        )
        if same_row and candidate_left >= right - 3:
            ranked.append(((0.0, candidate_left - right), candidate))
            continue
        horizontal_overlap = min(right, candidate_right) - max(left, candidate_left)
        vertical_gap = candidate_top - bottom
        if -2 <= vertical_gap <= height * 3 and (
            horizontal_overlap >= 0 or abs(candidate_left - left) <= height * 2
        ):
            ranked.append(((1.0, max(vertical_gap, 0.0)), candidate))
    return [candidate for _, candidate in sorted(ranked, key=lambda item: item[0])]


def _labeled_field(
    lines: Lines,
    labels: Iterable[str],
    parser: Callable[[str, dict[str, Any]], Field | None],
) -> Field | None:
    ordered_labels = sorted(labels, key=len, reverse=True)
    for index, line in enumerate(lines):
        text = str(line.get("text", ""))
        label = next(
            (label for label in ordered_labels if label.casefold() in text.casefold()),
            None,
        )
        if label is None:
            continue
        suffix = _inline_suffix(text, label)
        if suffix:
            parsed = parser(suffix, line)
            if parsed is not None:
                return parsed
        for candidate in _nearby_lines(lines, index):
            parsed = parser(str(candidate.get("text", "")), candidate)
            if parsed is not None:
                return parsed
    return None


def _text_parser(text: str, line: dict[str, Any]) -> Field | None:
    value = text.strip(" \t:：")
    return _field(value, line) if value else None


def _pattern_parser(
    pattern: re.Pattern[str],
) -> Callable[[str, dict[str, Any]], Field | None]:
    def parse(text: str, line: dict[str, Any]) -> Field | None:
        match = pattern.search(text)
        return _field(match.group(1), line, raw=match.group(0)) if match else None

    return parse


def _labeled_money(
    lines: Lines, labels: Iterable[str], business_context: dict[str, Any]
) -> Field | None:
    def parse(text: str, line: dict[str, Any]) -> Field | None:
        synthetic_line = dict(line, text=text)
        amounts = _amounts_in_line(synthetic_line, business_context, allow_plain=True)
        return amounts[0] if amounts else None

    return _labeled_field(lines, labels, parse)


def _trade_number(lines: Lines, labels: Iterable[str]) -> Field | None:
    def parse(text: str, line: dict[str, Any]) -> Field | None:
        match = TRADE_NO_RE.search(text)
        return _field(match.group(0), line, raw=match.group(0)) if match else None

    candidate = _labeled_field(lines, labels, parse)
    if candidate is not None:
        return candidate
    for line in lines:
        match = TRADE_NO_RE.search(str(line.get("text", "")))
        if match:
            return _field(match.group(0), line, raw=match.group(0))
    return None


def _currency_field(amount: Field | None) -> Field | None:
    if amount is None or amount.get("currency") is None:
        return None
    return {
        "value": amount["currency"],
        "confidence": amount["confidence"],
        "source_box": amount["source_box"],
        "source": amount.get("currency_source"),
    }


def _candidate_totals_from_amount(amount: Field | None) -> list[dict[str, Any]]:
    if amount is None:
        return []
    return [
        {
            "kind": "labeled_amount",
            "amount": amount["value"],
            "currency": amount.get("currency"),
            "component_count": 1,
            "source_boxes": [amount["source_box"]] if amount.get("source_box") else [],
        }
    ]


def _transaction_rows(
    lines: Lines, business_context: dict[str, Any]
) -> list[dict[str, Any]]:
    dates: list[tuple[int, Field]] = []
    amounts: list[tuple[int, Field]] = []
    for index, line in enumerate(lines):
        date = _date_from_text(str(line.get("text", "")), line)
        if date is not None:
            dates.append((index, date))
        if not any(
            marker in str(line.get("text", "")).casefold() for marker in TOTAL_MARKERS
        ):
            amounts.extend(
                (index, amount) for amount in _amounts_in_line(line, business_context)
            )

    rows: list[dict[str, Any]] = []
    used_amounts: set[int] = set()
    for date_index, date in dates:
        ranked: list[tuple[float, int, Field]] = []
        date_bounds = _bounds(lines[date_index])
        for amount_position, (amount_index, amount) in enumerate(amounts):
            if amount_position in used_amounts:
                continue
            if amount_index == date_index:
                ranked.append((0.0, amount_position, amount))
                continue
            amount_bounds = _bounds(lines[amount_index])
            if date_bounds is None or amount_bounds is None:
                continue
            date_center = (date_bounds[1] + date_bounds[3]) / 2
            amount_center = (amount_bounds[1] + amount_bounds[3]) / 2
            row_height = max(
                date_bounds[3] - date_bounds[1],
                amount_bounds[3] - amount_bounds[1],
                1.0,
            )
            distance = abs(date_center - amount_center)
            if distance <= row_height * 0.8:
                ranked.append((distance, amount_position, amount))
        if not ranked:
            continue
        _, amount_position, amount = min(ranked, key=lambda item: item[0])
        used_amounts.add(amount_position)

        row_indices = {date_index, amounts[amount_position][0]}
        description_parts: list[tuple[float, str, int]] = []
        date_bounds = _bounds(lines[date_index])
        if date_bounds is not None:
            date_center = (date_bounds[1] + date_bounds[3]) / 2
            date_height = max(date_bounds[3] - date_bounds[1], 1.0)
            for index, line in enumerate(lines):
                bounds = _bounds(line)
                if index in row_indices or bounds is None:
                    continue
                center = (bounds[1] + bounds[3]) / 2
                if (
                    abs(center - date_center)
                    <= max(date_height, bounds[3] - bounds[1]) * 0.8
                ):
                    description_parts.append(
                        (bounds[0], str(line.get("text", "")).strip(), index)
                    )
        description = " ".join(text for _, text, _ in sorted(description_parts) if text)
        source_boxes = [
            box for box in (date.get("source_box"), amount.get("source_box")) if box
        ]
        rows.append(
            {
                "date": date,
                "amount": amount,
                "description": description or None,
                "source_boxes": source_boxes,
            }
        )
    return rows


def _transaction_totals(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for transaction in transactions:
        amount = transaction["amount"]
        grouped.setdefault(amount.get("currency"), []).append(amount)
    totals: list[dict[str, Any]] = []
    for currency, amounts in sorted(grouped.items(), key=lambda item: item[0] or ""):
        total = sum((Decimal(amount["value"]) for amount in amounts), Decimal(0))
        totals.append(
            {
                "kind": "transaction_sum",
                "amount": f"{total:.2f}",
                "currency": currency,
                "component_count": len(amounts),
                "source_boxes": [
                    amount["source_box"]
                    for amount in amounts
                    if amount.get("source_box")
                ],
            }
        )
    return totals


def _commerce_fields(
    lines: Lines,
    context: dict[str, Any],
    *,
    amount_labels: Iterable[str],
    trade_labels: Iterable[str],
    order_labels: Iterable[str] = (),
) -> ExtractorResult:
    amount = _labeled_money(lines, amount_labels, context)
    fields: dict[str, Field | None] = {
        "amount": amount,
        "currency": _currency_field(amount),
        "payment_method": _labeled_field(
            lines, ("支付方式", "付款方式", "Payment Method", "Paid via"), _text_parser
        ),
        "status": _labeled_field(
            lines, ("交易状态", "狀態", "状态", "Status"), _text_parser
        ),
        "paid_date": _labeled_field(
            lines,
            ("支付时间", "付款时间", "交易时间", "Paid Date", "Payment Date"),
            _date_from_text,
        ),
        "transaction_id": _trade_number(lines, trade_labels),
        "order_id": _labeled_field(
            lines,
            order_labels,
            _pattern_parser(re.compile(r"([A-Za-z0-9][A-Za-z0-9-]{5,})")),
        )
        if tuple(order_labels)
        else None,
    }
    return (
        {name: value for name, value in fields.items() if value is not None},
        [],
        _candidate_totals_from_amount(amount),
        [],
    )


def _taobao(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    return _commerce_fields(
        lines,
        context,
        amount_labels=("实付款", "實付款", "订单金额", "訂單金額", "Amount Paid"),
        trade_labels=("支付宝交易号", "支付寶交易號", "Transaction ID"),
        order_labels=("订单编号", "訂單編號", "Order ID"),
    )


def _xianyu(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    return _commerce_fields(
        lines,
        context,
        amount_labels=("实付款", "實付款", "成交金额", "成交金額", "Amount Paid"),
        trade_labels=("支付宝交易号", "支付寶交易號", "Transaction ID"),
        order_labels=("订单号", "訂單號", "Order ID"),
    )


def _alipay(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    return _commerce_fields(
        lines,
        context,
        amount_labels=("实付金额", "實付金額", "订单金额", "訂單金額", "Amount"),
        trade_labels=(
            "支付宝交易号",
            "支付寶交易號",
            "流水号",
            "流水號",
            "Transaction ID",
        ),
    )


def _vendor_receipt(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    amount = _labeled_money(
        lines, ("Total", "Amount Paid", "合计", "合計", "总计", "總計"), context
    )
    fields: dict[str, Field | None] = {
        "receipt_id": _labeled_field(
            lines,
            ("Receipt ID", "Receipt No", "收据编号", "收據編號", "小票号"),
            _pattern_parser(re.compile(r"([A-Za-z0-9][A-Za-z0-9-]{2,})")),
        ),
        "paid_date": _labeled_field(
            lines,
            ("Paid Date", "Payment Date", "Date", "付款日期", "支付日期"),
            _date_from_text,
        ),
        "amount": amount,
        "currency": _currency_field(amount),
        "payment_method": _labeled_field(
            lines, ("Payment Method", "Paid via", "支付方式", "付款方式"), _text_parser
        ),
    }
    return (
        {name: value for name, value in fields.items() if value is not None},
        [],
        _candidate_totals_from_amount(amount),
        [],
    )


def _travel_approval(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    del context
    fields: dict[str, Field | None] = {
        "destination": _labeled_field(
            lines, ("Destination", "出差地点", "出差地點", "目的地"), _text_parser
        ),
        "start_date": _labeled_field(
            lines,
            ("Start Date", "Departure Date", "开始日期", "開始日期", "出发日期"),
            _date_from_text,
        ),
        "end_date": _labeled_field(
            lines,
            ("End Date", "Return Date", "结束日期", "結束日期", "返回日期"),
            _date_from_text,
        ),
        "approval_status": _labeled_field(
            lines, ("Approval Status", "审批状态", "審批狀態"), _text_parser
        ),
    }
    return (
        {name: value for name, value in fields.items() if value is not None},
        [],
        [],
        [],
    )


def _ride_or_transit(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    transactions = _transaction_rows(lines, context)
    fields: dict[str, Field] = {}
    warnings: list[str] = []
    if any(
        transaction["amount"].get("currency_token") == "$"
        for transaction in transactions
    ):
        warnings.append(
            "A bare $ currency symbol is ambiguous; provide business_context.currency"
        )
    return fields, transactions, _transaction_totals(transactions), warnings


@dataclass(frozen=True)
class ProfileAdapter:
    profile: str
    aliases: tuple[str, ...]
    support_markers: tuple[str, ...]
    minimum_support_markers: int
    required_fields: tuple[str, ...]
    extractor: Extractor
    version: str = "1.0.0"

    def support_matches(self, lines: Lines) -> list[str]:
        text = compact_text("\n".join(str(line.get("text", "")) for line in lines))
        return [
            marker for marker in self.support_markers if compact_text(marker) in text
        ]

    def supports(self, lines: Lines) -> bool:
        return len(self.support_matches(lines)) >= self.minimum_support_markers


_ADAPTERS = (
    ProfileAdapter(
        "taobao_order_detail",
        ("taobao",),
        ("淘宝", "交易成功", "实付款", "支付宝交易号"),
        2,
        ("amount", "payment_method", "transaction_id"),
        _taobao,
    ),
    ProfileAdapter(
        "xianyu_order_detail",
        ("xianyu",),
        ("闲鱼", "閒魚", "xianyu", "成交金额"),
        1,
        ("order_id", "amount", "payment_method"),
        _xianyu,
    ),
    ProfileAdapter(
        "alipay_payment_detail",
        ("alipay",),
        ("支付宝", "支付寶", "alipay", "实付金额", "流水号"),
        1,
        ("amount", "transaction_id"),
        _alipay,
    ),
    ProfileAdapter(
        "vendor_receipt",
        ("receipt",),
        ("receipt", "收据", "收據", "小票", "invoice"),
        1,
        ("receipt_id", "paid_date", "amount", "currency", "payment_method"),
        _vendor_receipt,
    ),
    ProfileAdapter(
        "travel_approval",
        ("travel",),
        ("travelapproval", "出差申请", "出差申請", "审批状态", "審批狀態"),
        1,
        ("destination", "start_date", "end_date", "approval_status"),
        _travel_approval,
    ),
    ProfileAdapter(
        "ride_payment",
        ("didi", "ride"),
        ("didi", "滴滴", "ridehistory", "行程记录", "行程記錄", "打车"),
        1,
        ("transactions", "candidate_totals"),
        _ride_or_transit,
    ),
    ProfileAdapter(
        "transit_payment",
        ("mtr", "transit"),
        ("mtr", "港铁", "港鐵", "transit", "地铁", "地鐵"),
        1,
        ("transactions", "candidate_totals"),
        _ride_or_transit,
    ),
)

PROFILE_ADAPTERS: dict[str, ProfileAdapter] = {
    adapter.profile: adapter for adapter in _ADAPTERS
}
PROFILE_ALIASES: dict[str, str] = {
    alias: adapter.profile
    for adapter in _ADAPTERS
    for alias in (adapter.profile, *adapter.aliases)
}


def available_profiles() -> tuple[str, ...]:
    return tuple(PROFILE_ADAPTERS)


def normalize_profile(profile: str) -> str:
    if profile in ("auto", "generic"):
        return profile
    try:
        return PROFILE_ALIASES[profile]
    except KeyError as error:
        raise ValueError(f"Unknown OCR profile: {profile}") from error


def infer_profile(lines: Lines, source_path: Path | None = None) -> str:
    filename = source_path.name.casefold() if source_path is not None else ""
    filename_hints = (
        (("xianyu", "闲鱼"), "xianyu_order_detail"),
        (("alipay", "支付宝"), "alipay_payment_detail"),
        (("taobao", "淘宝", "order"), "taobao_order_detail"),
        (("receipt", "invoice", "收据"), "vendor_receipt"),
        (("travel", "approval", "出差", "审批"), "travel_approval"),
        (("didi", "ride", "滴滴", "打车"), "ride_payment"),
        (("mtr", "transit", "metro", "港铁", "地铁"), "transit_payment"),
    )
    for tokens, profile in filename_hints:
        if any(token in filename for token in tokens) and PROFILE_ADAPTERS[
            profile
        ].supports(lines):
            return profile

    ranked = sorted(
        (
            (
                len(adapter.support_matches(lines)),
                -adapter.minimum_support_markers,
                adapter.profile,
            )
            for adapter in _ADAPTERS
            if adapter.supports(lines)
        ),
        reverse=True,
    )
    return ranked[0][2] if ranked else "generic"


def extract_fields(
    lines: Iterable[dict[str, Any]],
    profile: str,
    *,
    source_path: Path | None = None,
    business_context: dict[str, Any] | None = None,
    expected_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    line_list = list(lines)
    context = dict(business_context or {})
    requested = normalize_profile(profile)
    resolved = (
        infer_profile(line_list, source_path) if requested == "auto" else requested
    )
    if resolved == "generic":
        return {
            "profile": "generic",
            "adapter": None,
            "support_status": "unsupported",
            "fields": {},
            "transactions": [],
            "candidate_totals": [],
            "warnings": ["No registered adapter supports this OCR layout"],
            "expected_fields": list(expected_fields or ()),
            "missing_expected_fields": list(expected_fields or ()),
            "profile_check": "unsupported",
        }

    adapter = PROFILE_ADAPTERS[resolved]
    matches = adapter.support_matches(line_list)
    if not adapter.supports(line_list):
        return {
            "profile": resolved,
            "adapter": {
                "schema": ADAPTER_SCHEMA,
                "name": adapter.profile,
                "version": adapter.version,
                "registry_version": ADAPTER_REGISTRY_VERSION,
            },
            "support_status": "unsupported",
            "support_markers_found": matches,
            "fields": {},
            "transactions": [],
            "candidate_totals": [],
            "warnings": [f"Layout is unsupported by adapter {adapter.profile}"],
            "expected_fields": list(expected_fields or adapter.required_fields),
            "missing_expected_fields": list(expected_fields or adapter.required_fields),
            "profile_check": "unsupported",
        }

    fields, transactions, totals, warnings = adapter.extractor(line_list, context)
    expected = tuple(expected_fields or adapter.required_fields)
    present = set(fields)
    if transactions:
        present.add("transactions")
    if totals:
        present.add("candidate_totals")
    missing = [name for name in expected if name not in present]
    if missing:
        warnings.append("Missing expected fields: " + ", ".join(missing))

    return {
        "profile": resolved,
        "adapter": {
            "schema": ADAPTER_SCHEMA,
            "name": adapter.profile,
            "version": adapter.version,
            "registry_version": ADAPTER_REGISTRY_VERSION,
        },
        "support_status": "supported",
        "support_markers_found": matches,
        "fields": fields,
        "transactions": transactions,
        "candidate_totals": totals,
        "warnings": warnings,
        "expected_fields": list(expected),
        "missing_expected_fields": missing,
        "profile_check": "pass" if not missing else "review",
    }
