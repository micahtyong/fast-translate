# groq_translation.py
import json
from typing import Optional
import os

from decouple import config
from groq import Groq
from pydantic import BaseModel

# Set up the Groq client
client = Groq(api_key=config("GROQ_API_KEY"))

# Model for the translation
class Translation(BaseModel):
    text: str
    comments: Optional[str] = None


# Translate text using the Groq API
def groq_translate(query, from_language, to_language):
    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that translates text from {from_language} to {to_language}."
                           f"You will only reply with the translation text and nothing else in JSON."
                           f" The JSON object must use the schema: {json.dumps(Translation.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Translate '{query}' from {from_language} to {to_language}."
            }
        ],
        model="deepseek-r1-distill-llama-70b",
        temperature=0.2,
        max_tokens=4096,
        stream=False,
        response_format={"type": "json_object"},
    )
    # Return the translated text
    return Translation.model_validate_json(chat_completion.choices[0].message.content)

class Translator:
    def __init__(self):
        self.client = Groq(api_key=config("GROQ_API_KEY"))

    def translate(self, text, target_language):
        """Translate text to the target language using Groq"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Your task is to translate text to {target_language}. You must ONLY output the translation, with no explanations, thoughts, or additional text. Do not include any thinking process or reasoning. Just the translation."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=4096,
                stream=False,
            )
            # Clean up the response to remove any potential thinking or explanations
            translation = response.choices[0].message.content.strip()
            # Remove any content between <think> tags
            if "<think>" in translation:
                translation = translation.split("</think>")[-1].strip()
            return translation
        except Exception as e:
            return f"Translation error: {str(e)}"