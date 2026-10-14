from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from src.gate import factual_verifier as fv


def fake_wikidata_get_json(params):
    if params.get("action") == "wbsearchentities":
        return {"search": [{"id": "Q408", "label": "Australia", "description": "country in Oceania"}]}
    if params.get("action") == "wbgetentities" and params.get("props") == "claims|labels|descriptions":
        ids = params.get("ids", "").split("|")
        return {"entities": {
            item_id: {
                "descriptions": {"en": {"value": "country in Oceania"}},
                "claims": {
                    "P31": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q3624078"}}}}],
                    "P36": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q2817"}}}}],
                },
            }
            for item_id in ids
        }}
    if params.get("action") == "wbgetentities" and params.get("props") == "labels|aliases":
        ids = params.get("ids", "").split("|")
        return {"entities": {item_id: {"labels": {"en": {"value": "Canberra"}}, "aliases": {"en": [{"value": "Canberra"}]}} for item_id in ids}}
    raise AssertionError(f"Unexpected Wikidata request: {params}")


def test_percent_of_math_verified_without_network(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", lambda params: (_ for _ in ()).throw(AssertionError("network must not be used")))
    result = fv.verify_answer("What is 15% of 240?", "15% of 240 is 36.")
    assert result.status == "VERIFIED"
    assert "36" in result.evidence


def test_percent_of_math_wrong_answer_contradicted():
    result = fv.verify_answer("What is 15% of 240?", "15% of 240 is 35.")
    assert result.status == "CONTRADICTED"


def test_percent_of_math_conflicting_result_fails_closed():
    result = fv.verify_answer("What is 15% of 240?", "The answer is 35, but another result is 36.")
    assert result.status == "UNVERIFIED"


def test_capital_answer_verified_from_wikidata(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", fake_wikidata_get_json)
    result = fv.verify_answer("What is the capital of Australia?", "The capital of Australia is Canberra.")
    assert result.status == "VERIFIED"
    assert result.source_url == "https://www.wikidata.org/wiki/Q408"
    assert result.expected_values and "Canberra" in result.expected_values


def test_capital_evidence_labels_aliases_separately(monkeypatch):
    def fake_with_alias(params):
        payload = fake_wikidata_get_json(params)
        if params.get("props") == "labels|aliases":
            payload["entities"]["Q2817"]["aliases"]["en"].append({"value": "Canberra City"})
            payload["entities"]["Q2817"]["aliases"]["en"].append({"value": "canberra city"})
            payload["entities"]["Q2817"]["aliases"]["en"].append({"value": "CANBERRA"})
        return payload

    monkeypatch.setattr(fv, "_get_json", fake_with_alias)
    result = fv.verify_answer("What is the capital of Australia?", "The capital of Australia is Canberra.")
    assert result.status == "VERIFIED"
    assert result.evidence == (
        "Wikidata lists the current capital: Canberra. "
        "Alternate name(s)/aliases: Canberra City."
    )
    assert "current capital value(s)" not in result.evidence
    assert result.expected_values and "Canberra City" in result.expected_values
    alias_text = result.evidence.split("Alternate name(s)/aliases: ", 1)[1]
    assert alias_text == "Canberra City."
    assert set(("Canberra", "Canberra City", "canberra city", "CANBERRA")) <= set(result.expected_values)


def test_wrong_capital_contradicted_from_wikidata(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", fake_wikidata_get_json)
    result = fv.verify_answer("What is the capital of Australia?", "The capital of Australia is Sydney.")
    assert result.status == "CONTRADICTED"
    assert "Sydney" in result.evidence and "Canberra" in result.evidence


def test_correct_name_negated_is_not_accepted_as_verified(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", fake_wikidata_get_json)
    result = fv.verify_answer("What is the capital of Australia?", "Canberra is not the capital of Australia; Sydney is.")
    assert result.status != "VERIFIED"


def test_wrong_capital_claim_is_not_rescued_by_mentioning_correct_city(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", fake_wikidata_get_json)
    result = fv.verify_answer(
        "What is the capital of Australia?",
        "The capital of Australia is Sydney, although Canberra is also mentioned.",
    )
    assert result.status == "CONTRADICTED"


def test_list_of_multiple_capitals_is_not_silently_reduced_to_one(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", fake_wikidata_get_json)
    result = fv.verify_answer("What is the capital of Australia?", "The capital of Australia is Canberra and Sydney.")
    assert result.status == "CONTRADICTED"


def test_explicit_correction_can_be_verified(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", fake_wikidata_get_json)
    result = fv.verify_answer("What is the capital of Australia?", "The capital is not Sydney but Canberra.")
    # No country in the answer means this phrasing is not treated as a direct claim by our
    # deliberately narrow parser; it must fail closed rather than infer a relation.
    assert result.status == "UNVERIFIED"


def test_unsupported_question_fails_closed_without_network(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", lambda params: (_ for _ in ()).throw(AssertionError("network must not be used")))
    result = fv.verify_answer("Who is Isaac Newton?", "Isaac Newton was an English physicist.")
    assert result.status == "UNSUPPORTED"
    assert result.supported is False


def test_unavailable_source_fails_closed(monkeypatch):
    def fail(_params):
        raise TimeoutError("simulated timeout")
    monkeypatch.setattr(fv, "_get_json", fail)
    result = fv.verify_answer("What is the capital of Australia?", "Canberra")
    assert result.status == "UNAVAILABLE"
    assert result.supported is True


def _item_claim(item_id, *, rank="normal", end=None, start=None):
    claim = {
        "rank": rank,
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {"value": {"id": item_id}},
        },
    }
    qualifiers = {}
    if end is not None:
        qualifiers["P582"] = [{
            "snaktype": "value",
            "datavalue": {"value": {"time": end}},
        }]
    if start is not None:
        qualifiers["P580"] = [{
            "snaktype": "value",
            "datavalue": {"value": {"time": start}},
        }]
    if qualifiers:
        claim["qualifiers"] = qualifiers
    return claim


def _metadata_entity(description, type_id, capital_claims=()):
    return {
        "descriptions": {"en": {"value": description}},
        "claims": {
            "P31": [_item_claim(type_id)],
            "P36": list(capital_claims),
        },
    }


def test_country_resolution_rejects_continent_with_same_label(monkeypatch):
    def fake(params):
        if params["action"] == "wbsearchentities":
            return {"search": [
                {"id": "QCONT", "label": "Exampleland", "description": "continent"},
                {"id": "QCOUNTRY", "label": "Exampleland", "description": "country in Oceania"},
            ]}
        if params.get("props") == "claims|labels|descriptions":
            return {"entities": {
                "QCONT": _metadata_entity("continent", "Q5107"),
                "QCOUNTRY": _metadata_entity("country in Oceania", "Q3624078", [_item_claim("QCAP")]),
            }} if params["ids"] == "QCONT|QCOUNTRY" else {"entities": {
                "QCOUNTRY": _metadata_entity("country in Oceania", "Q3624078", [_item_claim("QCAP")]),
            }}
        if params.get("props") == "labels|aliases":
            return {"entities": {"QCAP": {"labels": {"en": {"value": "Example City"}}, "aliases": {"en": []}}}}
        raise AssertionError(params)

    monkeypatch.setattr(fv, "_get_json", fake)
    result = fv.verify_answer("What is the capital of Exampleland?", "Example City")
    assert result.status == "VERIFIED"
    assert result.source_url.endswith("QCOUNTRY")


def test_ambiguous_country_candidates_remain_unverified(monkeypatch):
    def fake(params):
        if params["action"] == "wbsearchentities":
            return {"search": [
                {"id": "Q1", "label": "Twinland", "description": "country in Europe"},
                {"id": "Q2", "label": "Twinland", "description": "country in Africa"},
            ]}
        if params.get("props") == "claims|labels|descriptions":
            return {"entities": {
                "Q1": _metadata_entity("country in Europe", "Q3624078"),
                "Q2": _metadata_entity("country in Africa", "Q3624078"),
            }}
        raise AssertionError(params)

    monkeypatch.setattr(fv, "_get_json", fake)
    result = fv.verify_answer("What is the capital of Twinland?", "A city")
    assert result.status == "UNVERIFIED"
    assert "Q1" in result.reason and "Q2" in result.reason


def test_missing_exact_country_match_is_unverified(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", lambda params: {"search": []})
    result = fv.verify_answer("What is the capital of Nowhere?", "A city")
    assert result.status == "UNVERIFIED"
    assert "No exact" in result.reason


def test_deprecated_and_historical_capitals_are_excluded(monkeypatch):
    def fake(params):
        if params["action"] == "wbsearchentities":
            return {"search": [{"id": "Q408", "label": "Australia", "description": "country in Oceania"}]}
        if params.get("props") == "claims|labels|descriptions":
            return {"entities": {"Q408": _metadata_entity(
                "country in Oceania",
                "Q3624078",
                [
                    _item_claim("QOLD", rank="deprecated"),
                    _item_claim("QOLD", end="+2000-00-00T00:00:00Z"),
                    _item_claim("Q2817"),
                ],
            )}}
        if params.get("props") == "labels|aliases":
            return {"entities": {"Q2817": {"labels": {"en": {"value": "Canberra"}}, "aliases": {"en": []}}}}
        raise AssertionError(params)

    monkeypatch.setattr(fv, "_get_json", fake)
    assert fv.verify_answer("What is the capital of Australia?", "Canberra").status == "VERIFIED"
    assert fv.verify_answer("What is the capital of Australia?", "Oldtown").status == "CONTRADICTED"


def test_only_historical_capital_claims_fail_closed(monkeypatch):
    def fake(params):
        if params["action"] == "wbsearchentities":
            return {"search": [{"id": "Q1", "label": "Oldland", "description": "country in Europe"}]}
        if params.get("props") == "claims|labels|descriptions":
            return {"entities": {"Q1": _metadata_entity(
                "country in Europe", "Q3624078",
                [_item_claim("QOLD", end="+2000-00-00T00:00:00Z")],
            )}}
        raise AssertionError(params)

    monkeypatch.setattr(fv, "_get_json", fake)
    result = fv.verify_answer("What is the capital of Oldland?", "Oldtown")
    assert result.status == "UNVERIFIED"
    assert "current capital" in result.reason


@pytest.mark.parametrize("failure", [
    TimeoutError("timeout"),
    ValueError("rate limited or malformed"),
])
def test_network_and_api_failures_are_unavailable(monkeypatch, failure):
    def fail(_params):
        raise failure
    monkeypatch.setattr(fv, "_get_json", fail)
    result = fv.verify_answer("What is the capital of Australia?", "Canberra")
    assert result.status == "UNAVAILABLE"
    assert result.supported is True


def test_malformed_search_payload_is_unavailable(monkeypatch):
    monkeypatch.setattr(fv, "_get_json", lambda params: {"search": {}})
    result = fv.verify_answer("What is the capital of Australia?", "Canberra")
    assert result.status == "UNAVAILABLE"
