import pickle
import os
import re



ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT_DIR, "models", "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(ROOT_DIR, "models", "mlb.pkl"), "rb") as f:
    mlb = pickle.load(f)



def clean_text(text):
    text = str(text).lower()

    
    text = re.sub(r"[^a-zA-Z,\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text



def extract_symptoms(text):
    text = clean_text(text)

    
    parts = [p.strip() for p in text.split(",")]

    symptoms = []

    for p in parts:
        if len(p) > 2:
            symptoms.append(p)

    return symptoms



def predict_disease(symptoms_list):

    if not symptoms_list:
        return {
            "main_disease": "Unknown",
            "main_confidence": 0.0,
            "top_3": [],
            "extracted_symptoms": []
        }

    
    symptoms_list = [
        clean_text(s)
        for s in symptoms_list
        if s and len(s.strip()) > 0
    ]

    
    known = set(mlb.classes_)

    filtered = [s for s in symptoms_list if s in known]

    
    if len(filtered) == 0:
        filtered = symptoms_list

    
    X = mlb.transform([filtered])

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    top_idx = probs.argsort()[-3:][::-1]

    return {
        "main_disease": classes[top_idx[0]],
        "main_confidence": float(probs[top_idx[0]]),

        "top_3": [
            {
                "disease": classes[i],
                "probability": float(probs[i])
            }
            for i in top_idx
        ],

        "extracted_symptoms": filtered
    }