import asyncio
import json
import re
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from agent_analyst.llm_client import LLMClient, ChatMessage
from .prompts import ARCHITECT_SYSTEM_PROMPT, ARCHITECT_USER_PROMPT
from .schemas import ArchitecturalAnalysis

class AgentArchitect:
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

    @staticmethod
    def _parse_json_with_fallback(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        cleaned = AgentArchitect._clean_json_text(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Tentative naïve de correction
        fixed = cleaned.replace("}\n{", "},\n{")
        return json.loads(fixed)

    def _unwrap_json(self, data: dict) -> dict:
        """
        Déballe le JSON s'il est encapsulé dans une clé racine.
        Cherche un dictionnaire qui contient les clés attendues de ArchitecturalAnalysis.
        """
        # Clés caractéristiques de ArchitecturalAnalysis
        required_keys = {"architecture", "tech_stack", "database", "consistency"}
        
        # 1. Vérifier si data est déjà le bon objet
        if any(k in data for k in required_keys):
            return data
            
        # 2. Chercher dans les valeurs de premier niveau
        for key, value in data.items():
            if isinstance(value, dict):
                # Si le sous-objet contient au moins une des clés requises
                if any(k in value for k in required_keys):
                    print(f"📦 JSON unwrapped from key: '{key}'")
                    return value
        
        # 3. Debug en cas d'échec
        print(f"⚠️ Impossible de déballer le JSON. Clés racine: {list(data.keys())}")
        return data

    async def a_generate_architecture(self, requirements_json: Dict[str, Any]) -> ArchitecturalAnalysis:
        messages = [
            ChatMessage(role="system", content=ARCHITECT_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=ARCHITECT_USER_PROMPT.format(requirements_json=json.dumps(requirements_json, ensure_ascii=False, indent=2))
            ),
        ]

        # On a besoin d'un contexte large pour l'architecture, donc 4096 tokens min si possible
        # Mistral 7B supporte souvent plus
        max_tokens = 4096 
        raw_response = await self.llm_client.acomplete(messages, max_tokens=max_tokens)

        content = (
            raw_response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ).strip()

        try:
            data = self._parse_json_with_fallback(content)
            data = self._unwrap_json(data)
        except Exception as e:
            print(f"❌ Erreur lors du parsing/validation: {e}")
            try:
                with open("debug_architecture_error.txt", "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"📄 Contenu brut sauvegardé dans 'debug_architecture_error.txt'")
            except Exception as write_err:
                print(f"⚠️ Impossible de sauvegarder le debug: {write_err}")

            return ArchitecturalAnalysis(
                summary="Échec du parsing JSON de l'architecture.",
                entities=[],
                architecture={"pattern": "monolith", "description": "Erreur parsing"},
                database={"schema_type": "sql", "content": "", "tables": []},
                tech_stack={"justification": "Erreur parsing"},
                consistency={"is_consistent": False, "issues": ["JSON invalide"]},
                complexity={"technical_score": 10, "dev_days_estimate": 0, "reasoning": "Erreur"},
                raw_model_output={"raw_text": content, "parse_error": str(e)},
            )

        analysis = ArchitecturalAnalysis(**data)
        analysis.raw_model_output = raw_response
        return analysis

    def generate_architecture(self, requirements_json: Dict[str, Any]) -> ArchitecturalAnalysis:
        return asyncio.run(self.a_generate_architecture(requirements_json))
