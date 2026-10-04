from neurocare import predictor


def test_empty_input_returns_unknown():
    result = predictor.predict_disease([])
    assert result["main_disease"] == "Unknown"
    assert result["top_3"] == []


def test_prediction_structure_and_probabilities():
    result = predictor.predict_disease(["headache", "vomiting", "fatigue"])
    assert len(result["top_3"]) == 3
    assert 0.0 <= result["main_confidence"] <= 1.0
    probs = [d["probability"] for d in result["top_3"]]
    assert probs == sorted(probs, reverse=True)


def test_parkinson_symptoms():
    result = predictor.predict_disease(["tremor", "stiffness", "resting"])
    assert result["main_disease"] == "Parkinson's"


def test_stroke_symptoms():
    result = predictor.predict_disease(["facial", "drooping", "slurred", "speech"])
    assert result["main_disease"] == "Stroke"
