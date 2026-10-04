from neurocare.chatbot import clean_json, normalize, parse_symptoms


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Severe HEADACHE!!  ") == "severe headache"


def test_clean_json_valid():
    assert clean_json('{"symptoms": ["fatigue"]}') == {"symptoms": ["fatigue"]}


def test_clean_json_extracts_json_from_noisy_text():
    raw = 'Sure! Here you go: {"symptoms": ["dizziness"]} hope it helps'
    assert clean_json(raw) == {"symptoms": ["dizziness"]}


def test_clean_json_invalid_returns_empty():
    assert clean_json("not json at all") == {"symptoms": []}
    assert clean_json("") == {"symptoms": []}


def test_parse_symptoms_handles_strings_and_dicts():
    data = {"symptoms": ["Fatigue", {"name": "Blurred Vision"}, "", {"name": ""}]}
    assert parse_symptoms(data) == ["fatigue", "blurred vision"]
