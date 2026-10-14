"""Fail-closed verifier for a deliberately narrow set of factual claims.

Supported claims are percentage-of arithmetic and capital questions backed by
Wikidata. This module is not a general fact checker.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_DEFAULT_USER_AGENT = (
    "LocalLLMReliabilityGate/0.6 "
    "(educational local project; low-volume verification)"
)
USER_AGENT = (os.environ.get("WIKIDATA_USER_AGENT") or _DEFAULT_USER_AGENT).strip()
REQUEST_TIMEOUT_SECONDS = 3.0
VERIFIER_VERSION = "capital_wikidata_percent_math_fail_closed_v2"

_COUNTRY_CLASS_IDS = {"Q6256", "Q3624078"}  # country, sovereign state
_NON_COUNTRY_DESCRIPTION_MARKERS = (
    "continent", "historical entity", "historic entity", "former country",
    "historical country", "fictional", "mythological", "legendary", "city",
    "town", "village", "region",
)
_COUNTRY_DESCRIPTION_RE = re.compile(
    r"\b(?:country|nation|sovereign state|republic|kingdom|federation)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationResult:
    status: str  # VERIFIED, CONTRADICTED, UNVERIFIED, UNSUPPORTED, UNAVAILABLE
    supported: bool
    reason: str
    evidence: str = ""
    source_url: str = ""
    expected_values: tuple[str, ...] = ()


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = text.replace("â€™", "'")
    return re.sub(r"[^\w%]+", " ", text).strip()


def _get_json(params: dict[str, str]) -> dict:
    query = urlencode(params)
    request = Request(
        f"{WIKIDATA_API}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError("Wikidata returned an invalid or error response")
    return payload


def _item_id_from_snak(snak: object) -> str | None:
    if not isinstance(snak, dict) or snak.get("snaktype") != "value":
        return None
    datavalue = snak.get("datavalue", {})
    value = datavalue.get("value", {}) if isinstance(datavalue, dict) else {}
    item_id = value.get("id") if isinstance(value, dict) else None
    return str(item_id) if item_id else None


def _claim_item_ids(claims: object, property_id: str) -> list[str]:
    if not isinstance(claims, dict):
        return []
    values: list[str] = []
    for claim in claims.get(property_id, []):
        if not isinstance(claim, dict):
            continue
        item_id = _item_id_from_snak(claim.get("mainsnak"))
        if item_id and item_id not in values:
            values.append(item_id)
    return values


def _entity_metadata(item_ids: list[str]) -> dict[str, dict]:
    if not item_ids:
        return {}
    payload = _get_json({
        "action": "wbgetentities",
        "ids": "|".join(item_ids),
        "props": "claims|labels|descriptions",
        "languages": "en",
        "format": "json",
    })
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError("Wikidata returned no entity metadata")
    return {
        str(item_id): entity
        for item_id, entity in entities.items()
        if isinstance(entity, dict) and "missing" not in entity
    }


def _is_country_candidate(description: str, type_ids: list[str]) -> bool:
    description_norm = _normalise(description)
    if any(marker in description_norm for marker in _NON_COUNTRY_DESCRIPTION_MARKERS):
        return False
    if _COUNTRY_CLASS_IDS.intersection(type_ids):
        return True
    return bool(_COUNTRY_DESCRIPTION_RE.search(description))


def _candidate_diagnostic(candidate: dict[str, object], accepted: bool) -> str:
    item_id = str(candidate.get("id", "?"))
    description = str(candidate.get("description") or "no description")
    type_ids = ",".join(str(value) for value in candidate.get("type_ids", ())) or "no P31"
    state = "country candidate" if accepted else "rejected as non-country/insufficient metadata"
    return f"{item_id} ({description}; P31={type_ids}; {state})"


def _search_exact_item(subject: str) -> tuple[str | None, str]:
    """Resolve one exact label only when Wikidata metadata identifies a country."""
    payload = _get_json({
        "action": "wbsearchentities", "search": subject, "language": "en",
        "uselang": "en", "type": "item", "limit": "10", "format": "json",
    })
    results = payload.get("search")
    if not isinstance(results, list):
        raise ValueError("Wikidata returned an invalid search response")

    exact: dict[str, dict[str, object]] = {}
    for item in results:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        if _normalise(str(item.get("label", ""))) != _normalise(subject):
            continue
        item_id = str(item["id"])
        exact.setdefault(item_id, {
            "id": item_id,
            "description": str(item.get("description") or ""),
            "type_ids": [],
        })
    if not exact:
        return None, f"No exact Wikidata subject match was found for {subject!r}."

    metadata = _entity_metadata(list(exact))
    candidates: list[dict[str, object]] = []
    for item_id, search_item in exact.items():
        entity = metadata.get(item_id)
        if not isinstance(entity, dict):
            candidates.append({
                "id": item_id,
                "description": "",
                "type_ids": [],
                "metadata_present": False,
            })
            continue
        descriptions = entity.get("descriptions", {})
        entity_description = (
            descriptions.get("en", {}).get("value")
            if isinstance(descriptions, dict) and isinstance(descriptions.get("en"), dict)
            else None
        )
        candidates.append({
            "id": item_id,
            "description": str(entity_description or search_item.get("description") or ""),
            "type_ids": _claim_item_ids(entity.get("claims", {}), "P31"),
            "metadata_present": True,
        })

    accepted = [
        candidate for candidate in candidates
        if _is_country_candidate(
            str(candidate.get("description", "")),
            list(candidate.get("type_ids", [])),
        ) and candidate.get("metadata_present", False)
    ]
    diagnostic = "; ".join(
        _candidate_diagnostic(candidate, candidate in accepted) for candidate in candidates
    )
    if len(accepted) == 1:
        return str(accepted[0]["id"]), f"Resolved country subject: {diagnostic}"
    if not accepted:
        return None, f"No exact-match Wikidata candidate was identified as a country: {diagnostic}"
    return None, f"Country subject is ambiguous; refusing to choose among: {diagnostic}"


def _entity_claims(item_id: str) -> dict:
    payload = _get_json({
        "action": "wbgetentities", "ids": item_id,
        "props": "claims|labels|descriptions", "languages": "en", "format": "json",
    })
    entities = payload.get("entities")
    entity = entities.get(item_id) if isinstance(entities, dict) else None
    if not isinstance(entity, dict) or "missing" in entity:
        raise ValueError("Subject item was not available in Wikidata")
    return entity


def _labels_for_items(item_ids: list[str]) -> dict[str, tuple[str, ...]]:
    if not item_ids:
        return {}
    payload = _get_json({
        "action": "wbgetentities", "ids": "|".join(item_ids),
        "props": "labels|aliases", "languages": "en", "format": "json",
    })
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError("Wikidata returned no capital labels")
    output: dict[str, tuple[str, ...]] = {}
    for item_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        names: list[str] = []
        labels = entity.get("labels", {})
        label = (
            labels.get("en", {}).get("value")
            if isinstance(labels, dict) and isinstance(labels.get("en"), dict) else None
        )
        if label:
            names.append(str(label))
        aliases = entity.get("aliases", {})
        for alias in aliases.get("en", []) if isinstance(aliases, dict) else []:
            value = alias.get("value") if isinstance(alias, dict) else None
            if value:
                names.append(str(value))
        output[str(item_id)] = tuple(sorted(set(names), key=len, reverse=True))
    return output


def _capital_question_subject(question: str) -> str | None:
    match = re.fullmatch(
        r"\s*(?:what is|what's|name)\s+(?:the\s+)?capital(?:\s+city)?\s+of\s+(.+?)\s*[?!.]?\s*",
        str(question or ""), flags=re.IGNORECASE,
    )
    if not match:
        return None
    subject = match.group(1).strip().strip(" .?!,")
    return subject or None


def _extract_capital_claims(answer: str, subject: str) -> list[tuple[str, bool]]:
    """Extract direct capital assertions as (value, is_negated) pairs."""
    escaped_subject = re.escape(subject)
    patterns = (
        rf"\bcapital(?:\s+city)?\s+of\s+{escaped_subject}\s+(?:is|was)\s+(?P<value>[^.!?;]+)",
        rf"\b(?P<value>[^.!?;,]+?)\s+(?:is|was)\s+(?P<negation>not\s+|never\s+)?(?:the\s+)?capital(?:\s+city)?\s+of\s+{escaped_subject}\b",
    )
    claims: list[tuple[str, bool]] = []
    for index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
            value = match.group("value").strip().strip(" .,!?:;")
            negated = bool(match.groupdict().get("negation"))
            if index == 0 and re.match(r"^(?:not|never)\s+", value, flags=re.IGNORECASE):
                negated = True
                value = re.sub(r"^(?:not|never)\s+", "", value, flags=re.IGNORECASE)
                contrast = re.split(r"\bbut\b", value, maxsplit=1, flags=re.IGNORECASE)
                if len(contrast) == 2:
                    negative_value = contrast[0].strip().strip(" .,!?:;")
                    positive_value = contrast[1].strip().strip(" .,!?:;")
                    if negative_value and (negative_value, True) not in claims:
                        claims.append((negative_value, True))
                    if positive_value and (positive_value, False) not in claims:
                        claims.append((positive_value, False))
                    continue
            value = re.split(
                r",|\b(?:but|however|although|because)\b", value,
                maxsplit=1, flags=re.IGNORECASE,
            )[0].strip().strip(" .,!?:;")
            if value and (value, negated) not in claims:
                claims.append((value, negated))

    direct = str(answer or "").strip().strip(" .,!?:;")
    if not claims and direct and len(direct.split()) <= 4 and not re.search(
        r"\b(?:don't know|do not know|not sure|unknown|unfamiliar|not provided)\b",
        direct, flags=re.IGNORECASE,
    ):
        claims.append((direct, False))
    return claims


def _parse_wikibase_time(snak: object) -> tuple[int, int, int] | None:
    if not isinstance(snak, dict) or snak.get("snaktype") != "value":
        return None
    datavalue = snak.get("datavalue", {})
    value = datavalue.get("value", {}) if isinstance(datavalue, dict) else {}
    raw_time = value.get("time") if isinstance(value, dict) else None
    if not isinstance(raw_time, str):
        return None
    match = re.match(r"^[+-](\d{1,6})-(\d{2})-(\d{2})T", raw_time)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return year, max(month, 1), max(day, 1)


def _claim_temporal_state(claim: object, today: date | None = None) -> str:
    """Return current, historical, future, or uncertain for one statement."""
    if not isinstance(claim, dict):
        return "uncertain"
    if claim.get("rank") in {"deprecated", 0}:
        return "historical"
    today_tuple = (today or date.today()).timetuple()[:3]
    qualifiers = claim.get("qualifiers", {})
    if not isinstance(qualifiers, dict):
        return "uncertain"

    end_snaks = qualifiers.get("P582", []) or []
    if not isinstance(end_snaks, list):
        return "uncertain"
    end_dates = [_parse_wikibase_time(snak) for snak in end_snaks]
    if any(value is None for value in end_dates):
        return "uncertain"
    if any(value <= today_tuple for value in end_dates if value is not None):
        return "historical"

    start_snaks = qualifiers.get("P580", []) or []
    if not isinstance(start_snaks, list):
        return "uncertain"
    start_dates = [_parse_wikibase_time(snak) for snak in start_snaks]
    if any(value is None for value in start_dates):
        return "uncertain"
    if any(value > today_tuple for value in start_dates if value is not None):
        return "future"
    return "current"


def _current_capital_ids(claims: object) -> tuple[list[str], bool]:
    """Return current preferred/normal P36 IDs and whether any claim was seen."""
    if not isinstance(claims, dict) or not isinstance(claims.get("P36", []), list):
        return [], False
    preferred: list[str] = []
    normal: list[str] = []
    saw_claim = False
    for claim in claims["P36"]:
        if not isinstance(claim, dict):
            continue
        saw_claim = True
        item_id = _item_id_from_snak(claim.get("mainsnak"))
        if not item_id or _claim_temporal_state(claim) != "current":
            continue
        target = preferred if claim.get("rank") == "preferred" else normal
        if item_id not in target:
            target.append(item_id)
    return preferred or normal, saw_claim


def _verify_capital(question: str, answer: str, subject: str) -> VerificationResult:
    try:
        subject_id, resolution_note = _search_exact_item(subject)
        if not subject_id:
            return VerificationResult("UNVERIFIED", True, resolution_note or "Subject is ambiguous.")

        subject_data = _entity_claims(subject_id)
        capital_ids, saw_claim = _current_capital_ids(subject_data.get("claims", {}))
        subject_url = f"https://www.wikidata.org/wiki/{subject_id}"
        if not capital_ids:
            reason = (
                f"Wikidata has no usable current capital (P36) statement for {subject}."
                if saw_claim else f"Wikidata has no usable capital (P36) statement for {subject}."
            )
            return VerificationResult("UNVERIFIED", True, reason, source_url=subject_url)

        target_names = _labels_for_items(capital_ids)
        expected = tuple(sorted(
            {name for names in target_names.values() for name in names},
            key=len, reverse=True,
        ))
        all_expected = ", ".join(sorted({name for names in target_names.values() for name in names}))
        if not expected:
            return VerificationResult(
                "UNVERIFIED", True,
                f"Wikidata returned current capital statements for {subject} but no usable English labels.",
                source_url=subject_url,
            )

        expected_norms = {_normalise(name) for name in expected}
        claims = _extract_capital_claims(answer, subject)
        affirmed = [value for value, negated in claims if not negated]
        contradictory = [value for value in affirmed if _normalise(value) not in expected_norms]
        matching = [value for value in affirmed if _normalise(value) in expected_norms]
        if contradictory:
            claimed = contradictory[0]
            return VerificationResult(
                "CONTRADICTED", True,
                f"The answer asserts a capital value ({claimed!r}) that conflicts with Wikidata for {subject}.",
                evidence=f"Claimed value: {claimed}. Wikidata current capital value(s): {all_expected}.",
                source_url=subject_url, expected_values=expected,
            )
        if matching:
            return VerificationResult(
                "VERIFIED", True,
                f"The answer's direct capital claim matches Wikidata for {subject}.",
                evidence=f"Wikidata lists the current capital value(s): {all_expected}.",
                source_url=subject_url, expected_values=expected,
            )
        return VerificationResult(
            "UNVERIFIED", True,
            "The answer did not make a clear affirmative capital claim that could be matched to the reference source.",
            evidence=f"Wikidata current capital value(s): {all_expected}.",
            source_url=subject_url, expected_values=expected,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError,
            IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return VerificationResult(
            "UNAVAILABLE", True,
            f"The independent Wikidata check could not complete ({type(exc).__name__}).",
            source_url="https://www.wikidata.org/",
        )


def _percent_of_expected(question: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%\s+of\s+(-?\d+(?:\.\d+)?)", question, flags=re.I)
    if not match:
        return None
    return float(match.group(1)) * float(match.group(2)) / 100.0


def _verify_percent_math(question: str, answer: str, expected: float) -> VerificationResult:
    claims = re.findall(r"(?:is|equals?)\s*([-+]?\d+(?:\.\d+)?)|=\s*([-+]?\d+(?:\.\d+)?)", answer, flags=re.I)
    values = [next(value for value in match if value) for match in claims]
    if not values:
        direct = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*%?\s*", answer)
        if direct:
            values = [direct.group(1)]
    if not values:
        return VerificationResult("UNVERIFIED", True, "No unambiguous numeric result was extracted for deterministic checking.")
    numbers = [float(value) for value in values]
    if any(not math.isclose(value, numbers[0], rel_tol=1e-9, abs_tol=1e-9) for value in numbers[1:]):
        return VerificationResult("UNVERIFIED", True, "The answer contains conflicting numeric results.", evidence=f"Expected result: {expected:g}.")
    actual = numbers[0]
    evidence = f"Deterministic calculation: percentage × base = {expected:g}; answer claims {actual:g}."
    if math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        return VerificationResult("VERIFIED", True, "The numeric answer matches the deterministic calculation.", evidence=evidence)
    return VerificationResult("CONTRADICTED", True, "The numeric answer does not match the deterministic calculation.", evidence=evidence)


def verify_answer(question: str, answer: str) -> VerificationResult:
    """Verify only supported claims; unsupported or unavailable cases fail closed."""
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    if not question or not answer:
        return VerificationResult("UNVERIFIED", False, "Question or answer is empty.")
    expected_percent = _percent_of_expected(question)
    if expected_percent is not None:
        return _verify_percent_math(question, answer, expected_percent)
    subject = _capital_question_subject(question)
    if subject:
        return _verify_capital(question, answer, subject)
    return VerificationResult(
        "UNSUPPORTED", False,
        "Independent verification is not implemented for this question type yet; the gate must VERIFY.",
    )
