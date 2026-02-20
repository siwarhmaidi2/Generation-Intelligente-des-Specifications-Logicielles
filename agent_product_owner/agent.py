import json
import re
from typing import Optional, Dict, Any, List

from agent_analyst.llm_client import LLMClient, ChatMessage
from .prompts import PRODUCT_OWNER_SYSTEM_PROMPT
from .schemas import ProductOwnerOutput, UserStory, Epic, Sprint, RoadmapItem


class AgentProductOwner:
    """
    Agent Product Owner : transforme des exigences JSON
    en un backlog Agile complet et structuré.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    # --------------------------------------------------
    # Point d'entrée principal
    # --------------------------------------------------
    async def generate_product_owner(
        self, requirements: Dict[str, Any]
    ) -> ProductOwnerOutput:
        user_prompt = self._build_user_prompt(requirements)

        messages = [
            ChatMessage(role="system", content=PRODUCT_OWNER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=4000)
        raw_content = response["choices"][0]["message"]["content"]

        return self._parse_response(raw_content)

    # --------------------------------------------------
    # Construction du prompt utilisateur
    # --------------------------------------------------
    @staticmethod
    def _build_user_prompt(requirements: Dict[str, Any]) -> str:
        return (
            "Voici les exigences du projet. Génère le backlog Agile complet "
            "en respectant strictement le format JSON demandé :\n\n"
            f"{json.dumps(requirements, ensure_ascii=False, indent=2)}"
        )

    # --------------------------------------------------
    # Parsing + normalisation + validation
    # --------------------------------------------------
    def _parse_response(self, raw_content: str) -> ProductOwnerOutput:

        cleaned = self._clean_json(raw_content)

        # ── Étape 1 : parsing JSON ────────────────────────────────────────────
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"❌ ERREUR JSON : {e}")
            print(f"   Ligne {e.lineno}, colonne {e.colno}")
            snippet = cleaned[max(0, e.pos - 80): e.pos + 80]
            print(f"   Extrait : ...{snippet!r}...")
            return self._empty_output()

        # ── Étape 2 : aplatir les US imbriquées dans les epics ───────────────
        flat_stories: list = data.get("user_stories", [])
        flat_story_ids = {s.get("id") for s in flat_stories}

        for epic in data.get("epics", []):
            nested = epic.get("user_stories", [])
            extracted_ids = []
            for item in nested:
                if isinstance(item, dict):
                    us_id = item.get("id")
                    if us_id and us_id not in flat_story_ids:
                        flat_stories.append(item)
                        flat_story_ids.add(us_id)
                    extracted_ids.append(us_id)
                else:
                    extracted_ids.append(item)
            epic["user_stories"] = [i for i in extracted_ids if i]

        data["user_stories"] = flat_stories

        # ── Étape 3 : normaliser les champs des User Stories ─────────────────
        FIBONACCI = {1, 2, 3, 5, 8, 13, 21}

        def nearest_fibonacci(n: int) -> int:
            return min(FIBONACCI, key=lambda x: abs(x - n))

        def normalize_priority(p: str) -> str:
            return {"must": "Must", "should": "Should", "could": "Could"}.get(
                p.lower(), "Should"
            )

        story_points_map: Dict[str, int] = {}

        for us in data.get("user_stories", []):
            sp = us.get("story_points", 3)
            if sp not in FIBONACCI:
                us["story_points"] = nearest_fibonacci(sp)
            us["priority"] = normalize_priority(us.get("priority", "Should"))
            us["risk"] = us.get("risk", "low").lower()
            us["business_value"] = max(1, min(10, int(us.get("business_value", 5))))
            if "dependencies" not in us:
                us["dependencies"] = []
            story_points_map[us["id"]] = us["story_points"]

        # ── Étape 4 : supprimer les dépendances circulaires ──────────────────
        data["user_stories"] = self._remove_circular_dependencies(
            data.get("user_stories", [])
        )

        # ── Étape 5 : recalculer capacity_points des sprints ─────────────────
        flat_sprints = []
        for sprint in data.get("sprints", []):
            us_in_sprint = sprint.get("user_stories", [])
            cleaned_ids = []
            for item in us_in_sprint:
                if isinstance(item, dict):
                    us_id = item.get("id")
                    if us_id and us_id not in flat_story_ids:
                        flat_stories.append(item)
                        flat_story_ids.add(us_id)
                        story_points_map[us_id] = item.get("story_points", 3)
                    cleaned_ids.append(us_id)
                else:
                    cleaned_ids.append(item)

            # Recalcule les vrais points du sprint
            real_points = sum(
                story_points_map.get(uid, 0) for uid in cleaned_ids if uid
            )

            flat_sprints.append({
                "id": sprint.get("id", ""),
                "user_stories": [i for i in cleaned_ids if i],
                "capacity_points": real_points,
            })

        data["sprints"] = flat_sprints

        # ── Étape 6 : validation Pydantic ─────────────────────────────────────
        try:
            output = ProductOwnerOutput(
                vision=data.get("vision", ""),
                epics=[Epic(**item) for item in data.get("epics", [])],
                user_stories=[
                    UserStory(**item) for item in data.get("user_stories", [])
                ],
                sprints=[Sprint(**item) for item in data.get("sprints", [])],
                roadmap=[
                    RoadmapItem(**item) for item in data.get("roadmap", [])
                ],
            )
            print("✅ Validation Pydantic réussie")
            return output

        except Exception as e:
            print(f"❌ ERREUR PYDANTIC : {e}")
            for key in ("epics", "user_stories", "sprints", "roadmap"):
                for i, item in enumerate(data.get(key, [])):
                    print(f"   [{key}][{i}] → {item}")
            return self._empty_output()

    # --------------------------------------------------
    # Suppression des dépendances circulaires (DFS)
    # --------------------------------------------------
    @staticmethod
    def _remove_circular_dependencies(
        user_stories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Détecte et supprime les dépendances circulaires entre US.
        Algorithme : DFS avec marquage visited/in_stack.
        La dépendance qui crée le cycle est retirée avec un warning.
        """
        graph: Dict[str, List[str]] = {
            us["id"]: list(us.get("dependencies", []))
            for us in user_stories
        }

        visited: set = set()
        in_stack: set = set()
        removed: List[tuple] = []

        def dfs(node: str, parent: str = "") -> None:
            if node not in graph:
                return
            visited.add(node)
            in_stack.add(node)

            deps_to_remove = []
            for dep in list(graph[node]):
                if dep not in graph:
                    continue
                if dep in in_stack:
                    # Cycle détecté → on retire cette dépendance
                    deps_to_remove.append(dep)
                    removed.append((node, dep))
                elif dep not in visited:
                    dfs(dep, node)

            for dep in deps_to_remove:
                graph[node].remove(dep)

            in_stack.discard(node)

        for us_id in list(graph.keys()):
            if us_id not in visited:
                dfs(us_id)

        if removed:
            print(f"⚠️  Dépendances circulaires supprimées :")
            for src, dst in removed:
                print(f"   {src} → {dst} (cycle retiré)")

        # Applique le graphe corrigé aux US
        id_to_us = {us["id"]: us for us in user_stories}
        for us_id, deps in graph.items():
            if us_id in id_to_us:
                id_to_us[us_id]["dependencies"] = deps

        return user_stories

    # --------------------------------------------------
    # Nettoyage du JSON brut (robuste pour Mistral)
    # --------------------------------------------------
    @staticmethod
    def _clean_json(text: str) -> str:
        # 1. Supprime les balises Markdown
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        text = re.sub(r"\r\n", "\n", text)

        # 2. Corrige les underscores mal échappés (bug Mistral)
        #    "user\_stories" → "user_stories"
        text = re.sub(r"\\_", "_", text)

        # 3. Isole le bloc JSON principal
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start: end + 1]

        # 4. Répare un JSON tronqué (coupé par max_tokens)
        try:
            json.loads(text)
        except json.JSONDecodeError:
            last_complete = text.rfind("},")
            if last_complete != -1:
                text = text[: last_complete + 1]
                opens = text.count("{") - text.count("}")
                arr_opens = text.count("[") - text.count("]")
                text += "]" * max(0, arr_opens) + "}" * max(0, opens)

        return text.strip()

    # --------------------------------------------------
    # Sortie vide en cas d'erreur critique
    # --------------------------------------------------
    @staticmethod
    def _empty_output() -> ProductOwnerOutput:
        return ProductOwnerOutput(
            vision="Erreur lors de la génération",
            epics=[],
            user_stories=[],
            sprints=[],
            roadmap=[],
        )