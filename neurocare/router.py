import logging
import os
import unicodedata
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.strip()


def classify(user_text, session_active=False):

    normalized = normalize(user_text)

    
    if session_active:
        return "DIAG"

    
    greetings = [
        "hello", "hi", "hey", "bonjour", "salut",
        "hola", "how are you", "ça va"
    ]

    if normalized in greetings:
        return "CHAT"

    

    models = [
            "llama-3.1-8b-instant"
        ]

    system_prompt = """
You are a medical conversation router.

Task:
Classify the user message into ONLY one label:

- DIAG: if the message contains symptoms, health problems, pain, neurological issues, or medical complaints.
- CHAT: if the message is greeting, small talk, or unrelated to health.

Return ONLY:
DIAG or CHAT
"""

    for model in models:
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0,
                max_tokens=5,
                timeout=8.0
            )

            if response and response.choices:
                result = response.choices[0].message.content.strip().upper()

                logger.debug("Router %s -> %s", model, result)

                if "DIAG" in result:
                    return "DIAG"
                if "CHAT" in result:
                    return "CHAT"

        except Exception as e:
            logger.warning("Router error with %s: %s", model, e)
            continue

    
    return "CHAT"