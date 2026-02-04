import asyncio
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

from .llm_client import LLMClient, ChatMessage
from .prompts import ANALYST_SYSTEM_PROMPT, ANALYST_SYSTEM_PROMPT_FAST
from .schemas import RequirementsAnalysis


class AgentAnalyst:
    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        load_dotenv()
        self.llm_client = llm_client or LLMClient()
    @staticmethod
    def _clean_json_text(text: str) -> str:
        """
        Nettoie le texte retourné par le LLM pour obtenir un JSON valide.
        Supprime les ```json et ``` ainsi que les caractères inutiles.
        """
        # Supprimer les blocs Markdown ```json ... ```
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        
        # Supprimer les retours à la ligne multiples et espaces inutiles
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"\s+", " ", text)

        # Prendre uniquement le texte entre { et } le plus large possible
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        return text.strip()

    # @staticmethod
    # def _clean_json_text(text: str) -> str:
    #     text = re.sub(r"```json\s*", "", text)
    #     text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    #     text = text.strip()
    #     start = text.find("{")
    #     end = text.rfind("}")
    #     if start != -1 and end != -1 and end > start:
    #         text = text[start : end + 1]
    #     return text.strip()
    

    @staticmethod
    def _fix_json_syntax(text: str) -> str:
        # Corrections basiques (virgules manquantes)
        text = re.sub(r"\}\s*\n\s*\{", "},\n{", text)
        text = re.sub(r"\]\s*\n\s*\{", "],\n{", text)
        return text

    @staticmethod
    def _parse_json_with_fallback(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        cleaned = AgentAnalyst._clean_json_text(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        fixed = AgentAnalyst._fix_json_syntax(cleaned)
        return json.loads(fixed)

    async def aanalyze_text(self, text: str) -> RequirementsAnalysis:
        mode = (os.getenv("ANALYST_MODE", "full") or "full").strip().lower()
        system_prompt = ANALYST_SYSTEM_PROMPT_FAST if mode == "fast" else ANALYST_SYSTEM_PROMPT

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "Voici le texte à analyser. Retourne UNIQUEMENT le JSON structuré demandé.\n\n"
                    f"{text}"
                ),
            ),
        ]

        max_tokens = int(os.getenv("LLM_MAX_TOKENS_FAST", "900")) if mode == "fast" else 2048
        raw_response = await self.llm_client.acomplete(messages, max_tokens=max_tokens)

        content = (
            raw_response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ).strip()

        try:
            data = self._parse_json_with_fallback(content)
        except Exception as e:
            return RequirementsAnalysis(
                summary="Échec du parsing JSON, voir raw_model_output.",
                raw_model_output={"raw_text": content, "api_response": raw_response, "parse_error": str(e)},
            )

        analysis = RequirementsAnalysis(**data)
        analysis.raw_model_output = raw_response
        return analysis

    def analyze_text(self, text: str) -> RequirementsAnalysis:
        return asyncio.run(self.aanalyze_text(text))



