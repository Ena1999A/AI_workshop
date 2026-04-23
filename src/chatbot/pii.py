"""
PII detection, masking, and customer lookup for Croatian leasing chatbot.

Detected fields: OIB, IBAN, email, phone, date of birth, license plate, contract number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_OIB = re.compile(r"\b\d{11}\b")
_IBAN = re.compile(r"\bHR\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{1}\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b(\+385|00385|0)[\s\-]?[1-9]\d[\s\-]?\d{3}[\s\-]?\d{3,4}\b")
_DATE = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\b")
_LICENSE_PLATE = re.compile(r"\b([A-ZŠĐŽČĆ]{2})\s?(\d{3,4})\s?([A-ZŠĐŽČĆ]{2})\b")
_CONTRACT = re.compile(r"\bUG-\d{4}-\d{5}\b")


def mask_pii(text: str) -> tuple[str, dict[str, object]]:
    """
    Detect and mask PII in text.
    Returns (masked_text, dict of found values keyed by field name).
    """
    found: dict[str, object] = {}

    for m in _OIB.finditer(text):
        found["oib"] = m.group()
    for m in _IBAN.finditer(text):
        found["iban"] = re.sub(r"\s", "", m.group())
    for m in _EMAIL.finditer(text):
        found["email"] = m.group()
    for m in _PHONE.finditer(text):
        found["telefon"] = m.group()
    for m in _DATE.finditer(text):
        found["datum_rodjenja"] = m.group()
    for m in _LICENSE_PLATE.finditer(text):
        found["registarska_oznaka"] = m.group()
    for m in _CONTRACT.finditer(text):
        found.setdefault("brojevi_ugovora", [])
        found["brojevi_ugovora"].append(m.group())  # type: ignore[union-attr]

    masked = text
    masked = _CONTRACT.sub("[BROJ_UGOVORA]", masked)
    masked = _IBAN.sub("[IBAN]", masked)
    masked = _OIB.sub("[OIB]", masked)
    masked = _EMAIL.sub("[EMAIL]", masked)
    masked = _PHONE.sub("[TELEFON]", masked)
    masked = _DATE.sub("[DATUM]", masked)
    masked = _LICENSE_PLATE.sub("[REG_OZNAKA]", masked)

    return masked, found


def identify_customer(
    pii_found: dict[str, object],
    customers_path: Path,
) -> Optional[tuple[str, dict]]:
    """
    Look up a customer from the PII registry.
    Returns (case_id, customer_dict) or None if no match.
    """
    if not customers_path.exists() or not pii_found:
        return None

    customers: dict = json.loads(customers_path.read_text(encoding="utf-8"))

    scalar_fields = ["oib", "iban", "email", "telefon", "registarska_oznaka"]

    for case_id, data in customers.items():
        for field in scalar_fields:
            if field in pii_found:
                if str(data.get(field, "")).lower() == str(pii_found[field]).lower():
                    return case_id, data

        if "brojevi_ugovora" in pii_found:
            for contract in pii_found["brojevi_ugovora"]:  # type: ignore[union-attr]
                if contract in data.get("brojevi_ugovora", []):
                    return case_id, data

    return None
