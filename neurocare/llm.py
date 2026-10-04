import logging
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.1-8b-instant"


class LLMHandler:

    def __init__(self):
        self._client = None

    @property
    def client(self):
        
        if self._client is None:
            self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self._client

   
    
    def _call(self, messages, temperature=0.5, max_tokens=200):

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ""

   
    # CHAT MODE 
    def ask(self, text):

        system = """
You are a neurological symptom checker assistant.

RULES:
- VERY SHORT answers (max 2 lines)
- NEVER give long explanations
- ALWAYS ask: "What other symptoms do you have?"
- If user says "I feel tired", respond ONLY:
  "I understand. What other symptoms do you have? (headache, dizziness, weakness)"
"""

        return self._call([
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ], temperature=0.3, max_tokens=80)

    
    
    def extract_medical_entities(self, text):

        system = """
You extract symptoms.

STRICT RULES:
- Return ONLY valid JSON
- No explanation
- No text outside JSON

FORMAT:
{
  "symptoms": ["fatigue", "dizziness"]
}

Convert user message into medical symptoms only.
"""

        raw = self._call([
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ], temperature=0.2, max_tokens=80)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return match.group(0)
        except re.error:
            pass

        return '{"symptoms": []}'

    
    
    def generate_medical_question(self, symptoms, disease):

        system = """
You are a medical assistant.

RULES:
- Ask ONLY ONE simple follow-up question
- Always ask about additional symptoms
- NEVER repeat previous questions
"""

        user = f"""
Symptoms: {symptoms}
Disease: {disease}

Ask ONE short follow-up question.
"""

        return self._call([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ], temperature=0.4, max_tokens=80)