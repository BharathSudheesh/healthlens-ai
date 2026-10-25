"""Text-based PDF blood report extraction and basic range flagging prototype."""
import re
from typing import Any

import fitz  # pip install pymupdf


TESTS = {
    "fasting blood glucose": ("Fasting blood glucose", "Glucose is a measure of blood sugar; interpretation depends on fasting status and clinical context."),
    "total cholesterol": ("Total cholesterol", "Total cholesterol is one part of assessing blood-fat levels and cardiovascular risk."),
    "ldl cholesterol": ("LDL cholesterol", "LDL cholesterol is one component of a lipid profile; targets depend on overall risk."),
    "hdl cholesterol": ("HDL cholesterol", "HDL cholesterol is one component of a lipid profile."),
    "triglycerides": ("Triglycerides", "Triglycerides are a type of fat measured in the blood."),
    "hemoglobin": ("Hemoglobin", "Hemoglobin is a protein in red blood cells that carries oxygen."),
    "creatinine": ("Creatinine", "Creatinine is commonly used with other information to assess kidney function."),
    "platelets": ("Platelets", "Platelets help blood clot; counts are interpreted with symptoms and other results."),
    "hba1c": ("HbA1c", "HbA1c reflects average blood sugar over roughly the previous 2–3 months."),
    "blood glucose": ("Blood glucose", "Glucose is a measure of blood sugar; interpretation depends on clinical context."),
    "glucose": ("Glucose", "Glucose is a measure of blood sugar; interpretation depends on clinical context."),
    "wbc": ("White blood cell count", "White blood cells are part of the immune system; counts can change for many reasons."),
}

# Fallbacks are only used if the PDF doesn't show a readable reference range.
# These are broad illustrative adult ranges and may not fit every person/lab.
FALLBACK_RANGES = {
    "hemoglobin": (12.0, 17.5, "g/dL"),
    "fasting blood glucose": (70.0, 99.0, "mg/dL"),
    "blood glucose": (70.0, 140.0, "mg/dL"),
    "glucose": (70.0, 140.0, "mg/dL"),
    "hba1c": (4.0, 5.6, "%"),
    "total cholesterol": (0.0, 199.0, "mg/dL"),
    "ldl cholesterol": (0.0, 99.0, "mg/dL"),
    "hdl cholesterol": (40.0, 9999.0, "mg/dL"),
    "triglycerides": (0.0, 149.0, "mg/dL"),
    "creatinine": (0.6, 1.3, "mg/dL"),
    "platelets": (150.0, 450.0, "10^3/µL"),
    "wbc": (4.0, 11.0, "10^3/µL"),
}


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text; scanned-image PDFs require OCR and aren't supported yet."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return "\n".join(page.get_text("text") for page in pdf)


def _status(value, low, high):
    if low is None or high is None:
        return "Range unavailable"
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Within range"


def _parse_test_line(line: str, key: str):
    number = r"[-+]?\d+(?:[.,]\d+)?"
    # Expected examples: "Glucose 145 mg/dL 70 - 99" or "HbA1c 6.1 % (4.0 - 5.6)"
    pattern = re.compile(
        rf"\b{re.escape(key)}\b\s*[:\-]?\s*({number})\s*([A-Za-zµ/%0-9^]+)?"
        rf"(?:\s*(?:\(|\[)?\s*(?:ref(?:erence)?(?:\s*range)?\s*[:=]?\s*)?"
        rf"({number})\s*(?:-|–|to)\s*({number})\s*(?:\)|\])?)?",
        re.IGNORECASE,
    )
    match = pattern.search(line)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    unit = (match.group(2) or "").strip()
    low = high = None
    if match.group(3) and match.group(4):
        low = float(match.group(3).replace(",", "."))
        high = float(match.group(4).replace(",", "."))
    return value, unit, low, high


def analyze_blood_report(pdf_bytes: bytes) -> dict[str, Any]:
    """Extract common test rows and compare with detected/fallback ranges."""
    try:
        text = extract_pdf_text(pdf_bytes)
    except Exception as exc:
        return {"text_found": False, "extracted_text": "", "results": [],
                "error": f"PDF could not be read ({type(exc).__name__})."}

    if not text.strip():
        return {"text_found": False, "extracted_text": "", "results": [],
                "error": "No selectable text was found in this PDF."}

    results, seen = [], set()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        normalized = re.sub(r"\s+", " ", line.lower())
        key = next((k for k in sorted(TESTS, key=len, reverse=True)
                    if re.search(r"\b" + re.escape(k) + r"\b", normalized)), None)
        if not key or key in seen:
            continue
        # PDF table cells may be extracted onto separate lines.
        candidate = " ".join(lines[index:index + 4])
        parsed = _parse_test_line(candidate, key) or _parse_test_line(line, key)
        if not parsed:
            continue

        value, unit, low, high = parsed
        ref_unit = unit
        if low is None or high is None:
            fallback = FALLBACK_RANGES.get(key)
            if fallback and (not unit or unit.lower() == fallback[2].lower()):
                low, high, ref_unit = fallback

        status = _status(value, low, high)
        label, base_explanation = TESTS[key]
        if status == "High":
            explanation = base_explanation + " This result is above the comparison range."
        elif status == "Low":
            explanation = base_explanation + " This result is below the comparison range."
        elif status == "Within range":
            explanation = base_explanation + " This result is within the comparison range."
        else:
            explanation = base_explanation + " The prototype could not confidently identify a reference range."

        results.append({
            "test": label,
            "value": value,
            "unit": unit,
            "result_display": f"{value:g} {unit}".strip(),
            "reference_low": low,
            "reference_high": high,
            "reference_display": f"{low:g}–{high:g} {ref_unit}".strip()
                if low is not None and high is not None else "Not confidently identified",
            "status": status,
            "explanation": explanation,
            "source_line": line.strip(),
        })
        seen.add(key)

    return {"text_found": True, "extracted_text": text, "results": results, "error": None}
