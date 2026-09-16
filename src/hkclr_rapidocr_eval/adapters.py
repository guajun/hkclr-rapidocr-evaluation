"""Business-aware OCR profile adapters and spatial extraction helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .schemas import ADAPTER_REGISTRY_VERSION, ADAPTER_SCHEMA

TRADE_NO_RE = re.compile(r"(?<!\d)20\d{26}(?!\d)")
DATE_RE = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:[-/.年])\s*(\d{1,2})\s*(?:[-/.月])\s*(\d{1,2})\s*日?"
    r"(?=\d{2}:\d{2}|\D|$)"
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
    if match is None:
        return None
    try:
        calendar_date(*(int(value) for value in match.groups()))
    except ValueError:
        return None
    return _field(_normalize_date(match), line, raw=match.group(0))


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
        currency = _currency_code(business_context["currency"])
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
        # Do not turn times, partial decimals, date fragments or identifiers into
        # money merely because a nearby label permits an unadorned number.
        start, end = match.span()
        if (start and text[start - 1] in ":/.") or (
            end < len(text) and text[end] in ":/."
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


def _standalone_money(
    text: str, line: dict[str, Any], context: dict[str, Any]
) -> Field | None:
    value = text.strip().replace("−", "-").replace("，", ",")
    yuan_suffix = value.endswith("元")
    if yuan_suffix:
        value = value[:-1].strip()
    match = AMOUNT_RE.fullmatch(value)
    if match is None:
        return None
    number = match.group("number").replace(",", "")
    if len(number.split(".")[0]) > 6:
        return None
    amount = _money_from_match(match, line, context)
    if amount is not None and yuan_suffix and amount.get("currency") is None:
        amount.update(currency="CNY", currency_token="元", currency_source="ocr")
    return amount


def _inline_suffix(text: str, label: str) -> str | None:
    start = text.casefold().find(label.casefold())
    if start < 0:
        return None
    suffix = text[start + len(label) :].lstrip(" \t:：")
    return suffix or None


def _contains_label(text: str, label: str) -> bool:
    if label.isascii():
        return re.search(r"(?<![A-Za-z])" + re.escape(label) + r"(?![A-Za-z])", text, re.IGNORECASE) is not None
    return label.casefold() in text.casefold()


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
            (label for label in ordered_labels if _contains_label(text, label)),
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
        return _standalone_money(text, line, business_context)

    # The label order expresses semantic priority (paid amount before order
    # total), irrespective of the engine's reading order across columns.
    for label in labels:
        if any(_contains_label(str(line.get("text", "")), label) for line in lines):
            return _labeled_field(lines, (label,), parse)
    return None


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
    # Phone screenshots wrap long trade numbers in the value column. Join only
    # adjacent digit-only fragments beneath the labeled value, never arbitrary
    # numbers elsewhere in the screenshot.
    for label_index, label_line in enumerate(lines):
        if not any(label in str(label_line.get("text", "")) for label in labels):
            continue
        for first in _nearby_lines(lines, label_index):
            digits = str(first.get("text", "")).strip()
            bounds = _bounds(first)
            if not re.fullmatch(r"20\d{8,25}", digits) or bounds is None:
                continue
            continuations = []
            for other in lines:
                tail = str(other.get("text", "")).strip()
                other_bounds = _bounds(other)
                if other is first or other_bounds is None or not tail.isdigit():
                    continue
                gap = other_bounds[1] - bounds[3]
                overlap = min(bounds[2], other_bounds[2]) - max(bounds[0], other_bounds[0])
                if 0 <= gap <= (bounds[3] - bounds[1]) * 2 and overlap > 0:
                    continuations.append((gap, tail, other))
            for _, tail, other in sorted(continuations, key=lambda item: item[0]):
                if TRADE_NO_RE.fullmatch(digits + tail):
                    value = _field(digits + tail, first, raw=digits + "\n" + tail)
                    value["confidence"] = min(_confidence(first), _confidence(other))
                    value["source_boxes"] = [_box(first), _box(other)]
                    return value
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
) -> tuple[list[dict[str, Any]], list[str]]:
    dates: list[tuple[int, Field]] = []
    amounts: list[tuple[int, Field]] = []
    for index, line in enumerate(lines):
        date = _date_from_text(str(line.get("text", "")), line)
        if date is not None:
            dates.append((index, date))
        if not any(
            marker in str(line.get("text", "")).casefold() for marker in TOTAL_MARKERS
        ):
            amount = _standalone_money(str(line.get("text", "")), line, business_context)
            if amount is not None and not amount.get("currency_token") and re.fullmatch(
                r"(?:19|20)\d{2}", str(line.get("text", "")).strip()
            ):
                amount = None
            if amount is not None:
                amounts.append((index, amount))
            elif date is not None:
                # Some engines put the date and an explicitly marked price in
                # one box; standalone integers need a separate spatial column.
                amounts.extend(
                    (index, amount)
                    for amount in _amounts_in_line(line, business_context)
                    if amount.get("currency_token")
                )

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_amounts: set[int] = set()
    ride_cards = any("下单时间" in str(line.get("text", "")) for line in lines)
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
            if amount_bounds[3] - amount_bounds[1] < (date_bounds[3] - date_bounds[1]) * 0.6:
                continue
            distance = abs(date_center - amount_center)
            if distance <= row_height * 0.8 and amount_bounds[0] >= date_bounds[2] - 3:
                ranked.append((distance, amount_position, amount))
            elif ride_cards and "下单时间" in str(lines[date_index].get("text", "")):
                # Ride fares precede the order date vertically. The previous
                # card bounds this search so neighboring fares cannot leak in.
                previous_bottom = max(
                    (bounds[3] for index, _ in dates if index != date_index
                     if (bounds := _bounds(lines[index])) is not None
                     and bounds[3] < date_bounds[1]),
                    default=0.0,
                )
                if (amount_bounds[0] >= date_bounds[2] - 3
                    and previous_bottom <= amount_bounds[1] < date_bounds[1]
                    and date_bounds[1] - amount_bounds[3] <= row_height * 5):
                    ranked.append((distance, amount_position, amount))
        if not ranked:
            warnings.append(f"Transaction date {date['value']} has no associated amount")
            continue
        if len(ranked) > 1:
            warnings.append(f"Transaction date {date['value']} has ambiguous amount candidates")
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
    if dates and len(rows) != len(dates):
        warnings.append(f"Incomplete transaction coverage: {len(rows)} of {len(dates)} dated rows")
    dated_bounds = [_bounds(lines[index]) for index, _ in dates]
    dated_bounds = [bounds for bounds in dated_bounds if bounds is not None]
    if dated_bounds:
        minimum_right = min(bounds[2] for bounds in dated_bounds)
        minimum_height = min(bounds[3] - bounds[1] for bounds in dated_bounds)
        unmatched = [position for position, (index, _) in enumerate(amounts)
                     if position not in used_amounts
                     if (bounds := _bounds(lines[index])) is not None
                     and bounds[0] >= minimum_right - 3
                     and bounds[3] - bounds[1] >= minimum_height * 0.6]
        if unmatched:
            warnings.append(f"Unassociated transaction amount candidates: {len(unmatched)}")
    return rows, warnings


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
    paid_date_labels: Iterable[str] = (
        "支付时间", "付款时间", "Paid Date", "Payment Date",
    ),
) -> ExtractorResult:
    order_labels = tuple(order_labels)
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
            lines, paid_date_labels, _date_from_text,
        ),
        "order_date": _labeled_field(
            lines,
            ("创建时间", "創建時間", "下单时间", "下單時間", "Order Date"),
            _date_from_text,
        ) if order_labels else None,
        "transaction_id": _trade_number(lines, trade_labels),
        "order_id": _labeled_field(
            lines,
            order_labels,
            _pattern_parser(re.compile(r"([A-Za-z0-9][A-Za-z0-9-]{5,})")),
        )
        if order_labels
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
        amount_labels=("实付款", "實付款", "成交金额", "成交金額", "成交价", "成交價", "Amount Paid"),
        trade_labels=("支付宝交易号", "支付寶交易號", "Transaction ID"),
        order_labels=("订单编号", "訂單編號", "订单号", "訂單號", "Order ID"),
    )


def _alipay(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    context = dict(context)
    unit_line = next((line for line in lines if re.search(
        r"单位\s*[:：]\s*元", str(line.get("text", ""))
    )), None)
    if "currency" not in context and unit_line is not None:
        context["currency"] = "CNY"
    fields, transactions, totals, warnings = _commerce_fields(
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
        paid_date_labels=(
            "支付时间", "付款时间", "交易时间", "时间：", "Paid Date", "Payment Date",
        ),
    )
    if unit_line is not None and fields.get("currency", {}).get("value") == "CNY":
        fields["currency"] = dict(_field("CNY", unit_line), source="ocr")
        fields["amount"]["currency_source"] = "ocr"
    return fields, transactions, totals, warnings


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
    table = _approval_table(lines)
    if table is not None:
        approvals, warnings = table
        return {"approvals": _approval_list_field(approvals)}, [], [], warnings
    fields: dict[str, Field | None] = {
        "destination": _labeled_field(
            lines, ("Destination", "出差地点", "出差地點", "目的地"), _text_parser
        ),
        "start_date": _labeled_field(
            lines,
            ("Start Date", "Start Time", "Departure Date", "开始日期", "開始日期", "出发日期"),
            _date_from_text,
        ),
        "end_date": _labeled_field(
            lines,
            ("End Date", "End Time", "Return Date", "结束日期", "結束日期", "返回日期"),
            _date_from_text,
        ),
        "approval_status": _labeled_field(
            lines, ("Approval Status", "审批状态", "審批狀態"), _text_parser
        ),
    }
    present = {name: value for name, value in fields.items() if value is not None}
    if len(present) == 4:
        present["approvals"] = _approval_list_field([dict(present)])
    return (
        present,
        [],
        [],
        [],
    )


def _approval_list_field(approvals: list[dict[str, Field]]) -> Field:
    return {
        "value": approvals,
        "confidence": min(
            (field["confidence"] for row in approvals for field in row.values()),
            default=0.0,
        ),
        "source_box": None,
    }


def _approval_table(lines: Lines) -> tuple[list[dict[str, Field]], list[str]] | None:
    headers = {}
    for line in lines:
        text = compact_text(str(line.get("text", "")))
        if "destination" in text or text in {"目的地", "出差地点", "出差地點"}:
            headers["destination"] = line
        if "starttime" in text and "endtime" in text:
            headers["dates"] = line
        if text in {"审批状态", "審批狀態", "approvalstatus"}:
            headers["approval_status"] = line
    if set(headers) != {"destination", "dates", "approval_status"}:
        return None
    header_bounds = {name: _bounds(line) for name, line in headers.items()}
    if any(bounds is None for bounds in header_bounds.values()):
        return None
    rows = []
    warnings = []
    dates_bounds = header_bounds["dates"]
    for line in lines:
        bounds = _bounds(line)
        if bounds is None or bounds[1] <= dates_bounds[3]:
            continue
        if min(bounds[2], dates_bounds[2]) <= max(bounds[0], dates_bounds[0]):
            continue
        matches = list(DATE_RE.finditer(str(line.get("text", ""))))
        if not matches:
            continue
        if len(matches) != 2 or any(_date_from_text(m.group(0), line) is None for m in matches):
            warnings.append("Approval table date interval is incomplete or ambiguous")
            continue
        row = {
            "start_date": _field(_normalize_date(matches[0]), line, raw=matches[0].group(0)),
            "end_date": _field(_normalize_date(matches[1]), line, raw=matches[1].group(0)),
        }
        for name in ("destination", "approval_status"):
            column = header_bounds[name]
            candidates = []
            for other in lines:
                other_bounds = _bounds(other)
                if other_bounds is None or other_bounds[1] <= column[3]:
                    continue
                overlap = min(column[2], other_bounds[2]) - max(column[0], other_bounds[0])
                distance = abs((bounds[1] + bounds[3] - other_bounds[1] - other_bounds[3]) / 2)
                height = max(bounds[3] - bounds[1], other_bounds[3] - other_bounds[1])
                if overlap > 0 and distance <= height:
                    candidates.append(other)
            if len(candidates) == 1:
                row[name] = _field(str(candidates[0]["text"]).strip(), candidates[0])
            else:
                warnings.append(f"Approval table row has missing or ambiguous {name}")
        if len(row) == 4:
            if row["start_date"]["value"] > row["end_date"]["value"]:
                warnings.append("Approval date interval ends before it starts")
            rows.append(row)
    if not rows:
        warnings.append("No complete approval rows were extracted")
    status_column = header_bounds["approval_status"]
    status_count = sum(
        1 for line in lines if (bounds := _bounds(line)) is not None
        and bounds[1] > status_column[3]
        and min(status_column[2], bounds[2]) > max(status_column[0], bounds[0])
    )
    if status_count != len(rows):
        warnings.append(f"Incomplete approval coverage: {len(rows)} of {status_count} status rows")
    return rows, warnings


def _ride_or_transit(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    transactions, warnings = _transaction_rows(lines, context)
    fields: dict[str, Field] = {}
    if any(
        transaction["amount"].get("currency") is None
        for transaction in transactions
    ):
        warnings.append(
            "Transaction currency is unknown; provide business_context.currency"
        )
    return fields, transactions, _transaction_totals(transactions), warnings


def _transit(lines: Lines, context: dict[str, Any]) -> ExtractorResult:
    fields, transactions, totals, warnings = _ride_or_transit(lines, context)
    if any(row["amount"].get("explicit_sign") is None for row in transactions):
        warnings.append("Transit amount has no explicit debit/credit sign; verify payment direction")
    return fields, transactions, totals, warnings


@dataclass(frozen=True)
class ProfileAdapter:
    profile: str
    aliases: tuple[str, ...]
    support_markers: tuple[str, ...]
    minimum_support_markers: int
    required_fields: tuple[str, ...]
    extractor: Extractor
    version: str = "1.1.1"

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
        ("order_id", "amount"),
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
        ("approvals",),
        _travel_approval,
    ),
    ProfileAdapter(
        "ride_payment",
        ("didi", "ride"),
        ("didi", "滴滴", "ridehistory", "行程记录", "行程記錄", "打车", "呼叫返程", "再来一单"),
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
        _transit,
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
    trusted_transit = (
        requested == "transit_payment"
        and str(context.get("provider", "")).casefold() in {"mtr", "octopus"}
        and any(_date_from_text(str(line.get("text", "")), line) for line in line_list)
    )
    if trusted_transit:
        matches.append("business_context.provider=" + str(context["provider"]).casefold())
    if not adapter.supports(line_list) and not trusted_transit:
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
    present = {name for name, field in fields.items() if field.get("value") not in (None, [], "")}
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
        "profile_check": "pass" if not missing and not warnings else "review",
    }
