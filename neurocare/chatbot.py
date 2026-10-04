import json
import logging
import re
from dotenv import load_dotenv

from neurocare import router
from neurocare import predictor
from neurocare.llm import LLMHandler

load_dotenv()

logger = logging.getLogger(__name__)

ia = LLMHandler()



symptoms_storage = []


# CLEAN FUNCTION


def normalize(text):

    text = str(text).lower().strip()

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# JSON PARSER


def parse_symptoms(data):

    raw = data.get("symptoms", [])

    symptoms = []

    for item in raw:

        if isinstance(item, str):

            s = normalize(item)

            if s:
                symptoms.append(s)

        elif isinstance(item, dict):

            name = normalize(item.get("name", ""))

            if name:
                symptoms.append(name)

    return symptoms


def clean_json(raw):

    if not raw:
        return {"symptoms": []}

    try:
        return json.loads(raw)

    except json.JSONDecodeError:

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:

            try:
                return json.loads(match.group(0))

            except json.JSONDecodeError:
                return {"symptoms": []}

    return {"symptoms": []}


# MAIN PIPELINE


def handle_user_input(user_text):

    global symptoms_storage

    text = user_text.lower().strip()

    

    decision = router.classify(
        user_text,
        session_active=len(symptoms_storage) > 0
    )

    
    # CHAT MODE
    

    if decision == "CHAT":

        return ia.ask(user_text)

    
    # EXTRACT SYMPTOMS
    

    raw = ia.extract_medical_entities(user_text)

    logger.debug("LLM raw extraction: %s", raw)

    data = clean_json(raw)

    symptoms_list = parse_symptoms(data)

    
    # STORE SYMPTOMS
    

    for sym in symptoms_list:

        sym = normalize(sym)

        if sym and sym not in symptoms_storage:

            symptoms_storage.append(sym)

            logger.debug("Symptom added: %s", sym)

    
    # FIRST FOLLOW-UP
    

    if len(symptoms_storage) == 1:

        return "I understand. What other symptoms do you have?"

    
    # ML PREDICTION
    

    analysis = predictor.predict_disease(
        symptoms_storage
    )

    main = analysis["main_disease"]

    conf = analysis["main_confidence"]

    top3 = analysis["top_3"]

    logger.debug("Predicted disease: %s", main)
    logger.debug("Confidence: %.3f", conf)

    
    # FINAL DIAGNOSIS
    

    if (
        len(symptoms_storage) >= 3
        or "finish" in text
        or "diagnostic" in text
    ):

        # build top conditions
        top_text = "\n".join([

            f"- {d['disease']}"

            for d in top3
        ])

        # explanation prompt
        explanation_prompt = f"""
You are a neurological assistant.

Explain these neurological predictions simply.

Symptoms:
{", ".join(symptoms_storage)}

Top conditions:
{top_text}

Main prediction:
{main}

Rules:
- short explanation
- simple language
- no emotional text
- no certainty
- DO NOT ask another question
- DO NOT continue the conversation
- finish after explanation
"""

        explanation = ia.ask(explanation_prompt)

        # final response
        result = f"""
🧠 Neurological Analysis

Symptoms:
{", ".join(symptoms_storage)}

Top 3 possible conditions:
- {top3[0]['disease']}
- {top3[1]['disease']}
- {top3[2]['disease']}

Main suspected condition:
{main}

Explanation:
{explanation}
"""

        # reset session
        symptoms_storage.clear()

        return result

    
    # CONTINUE FLOW
    

    return "What other symptoms do you have?"