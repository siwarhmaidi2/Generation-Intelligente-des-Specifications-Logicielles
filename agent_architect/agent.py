import asyncio
import json
import re
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
from agent_analyst.llm_client import LLMClient, ChatMessage
from .prompts import ARCHITECT_SYSTEM_PROMPT, ARCHITECT_USER_PROMPT

# -------------------------
# Schemas
# -------------------------
class UMLDiagram(BaseModel):
    diagram_type: str
    content: str
    description: Optional[str] = None

class Entity(BaseModel):
    name: str
    attributes: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)

class DatabaseTable(BaseModel):
    name: str
    columns: List[str]

class DatabaseSchema(BaseModel):
    schema_type: str
    content: str
    tables: List[DatabaseTable] = Field(default_factory=list)

class TechStack(BaseModel):
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    database: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)
    justification: str = ""

class ArchitectureProposal(BaseModel):
    pattern: str
    description: str
    diagram: Optional[UMLDiagram] = None

class ModuleDependency(BaseModel):
    module_from: str
    module_to: str
    description: Optional[str] = None

class ConsistencyCheck(BaseModel):
    is_consistent: bool
    issues: List[str] = Field(default_factory=list)

class ComplexityEstimation(BaseModel):
    technical_score: int
    dev_days_estimate: int
    reasoning: str

class ArchitecturalAnalysis(BaseModel):
    summary: str
    entities: List[Entity]
    architecture: ArchitectureProposal
    uml_diagrams: List[UMLDiagram]
    database: DatabaseSchema
    dependencies: List[ModuleDependency]
    tech_stack: TechStack
    consistency: ConsistencyCheck
    complexity: ComplexityEstimation
    raw_model_output: Optional[dict] = None

# -------------------------
# Agent Architect avec LLM pour les attributs
# -------------------------
class AgentArchitect:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    @staticmethod
    def _clean_json(text: str) -> str:
        """
        Nettoie le texte JSON renvoyé par le LLM pour le rendre parseable par json.loads.
        - Supprime les blocs Markdown ```json ... ```
        - Remplace les backslashes non échappés
        - Supprime les retours à la ligne inutiles
        """
        # Supprimer ```json ... ```
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)

        # Supprimer retours à la ligne multiples
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\s*\n\s*", "\n", text)

        # Remplacer les backslashes non échappés par double backslash
        # Exemple: "C:\Users\..." -> "C:\\Users\\..."
        text = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r"\\\\", text)

        # Extraire uniquement le JSON entre { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return text.strip()

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Tente de parser le JSON, applique un nettoyage si nécessaire.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = AgentArchitect._clean_json(text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e2:
                # Fallback naïf : remplacer quotes simples par doubles
                fixed = cleaned.replace("'", '"')
                return json.loads(fixed)

    # Utiliser le LLM pour détecter les entités et attributs
    async def detect_entities_with_llm(self, requirements_json: Dict[str, Any]) -> List[Entity]:
        prompt = f"""
        Voici les exigences du projet :
        {json.dumps(requirements_json, ensure_ascii=False, indent=2)}

        Identifie les entités principales, leurs attributs et relations. 
        Renvoie uniquement un JSON au format :
        {{
            "entities": [
                {{
                    "name": "NomEntite",
                    "attributes": ["attr1", "attr2"],
                    "relations": ["AutreEntite"]
                }}
            ]
        }}
        """
        messages = [
            ChatMessage(role="system", content=ARCHITECT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=2048)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        data = self._parse_json(content)
        entities = [Entity(**e) for e in data.get("entities", [])]
        return entities

    # Génération UML classique
    @staticmethod
    def generate_uml(entities: List[Entity], requirements_json: Dict[str, Any]) -> List[UMLDiagram]:
        class_diagram = "@startuml\n"
        for e in entities:
            class_diagram += f"class {e.name} {{\n"
            for attr in e.attributes:
                class_diagram += f"  {attr}: string\n"
            class_diagram += "}\n"
        for i in range(len(entities)-1):
            class_diagram += f"{entities[i].name} --> {entities[i+1].name}\n"
        class_diagram += "@enduml"

        use_case_diagram = "@startuml\n"
        actors = [a.get("name", f"Actor{i}") for i, a in enumerate(requirements_json.get("actors", []))]
        for a in actors:
            use_case_diagram += f"actor {a}\n"
        for fr in requirements_json.get("functional_requirements", []):
            use_case_diagram += f"usecase \"{fr['description']}\" as {fr['id']}\n"
            for a in actors:
                use_case_diagram += f"{a} --> {fr['id']}\n"
        use_case_diagram += "@enduml"

        seq_diagram = "@startuml\n"
        if actors and entities and requirements_json.get("functional_requirements"):
            seq_diagram += f"{actors[0]} -> {entities[0].name}: {requirements_json['functional_requirements'][0]['description']}\n"
            seq_diagram += f"{entities[0].name} --> {actors[0]}: response\n"
        seq_diagram += "@enduml"

        return [
            UMLDiagram(diagram_type="class", content=class_diagram, description="Diagramme de classes"),
            UMLDiagram(diagram_type="use_case", content=use_case_diagram, description="Diagramme Use Case"),
            UMLDiagram(diagram_type="sequence", content=seq_diagram, description="Diagramme de séquence")
        ]

    # Détection dépendances
    @staticmethod
    def detect_dependencies(entities: List[Entity]) -> List[ModuleDependency]:
        return [ModuleDependency(module_from=entities[i].name,
                                 module_to=entities[i+1].name,
                                 description="Dépendance détectée")
                for i in range(len(entities)-1)]

    # Génération DB avec attributs détectés par LLM
   # Génération DB avec attributs détectés par LLM
    @staticmethod
    def generate_database(entities: List[Entity]) -> DatabaseSchema:
        tables = []
        content = ""
        for e in entities:
            columns = []
            for attr in e.attributes:
                if attr.lower() == "id":
                    columns.append(f"{attr} INT PRIMARY KEY")
                else:
                    columns.append(f"{attr} VARCHAR")
            # Si aucune colonne "id", on l'ajoute automatiquement
            if not any(c.lower().startswith("id") for c in columns):
                columns.insert(0, "id INT PRIMARY KEY")

            tables.append(DatabaseTable(name=e.name.lower(), columns=columns))
            content += f"CREATE TABLE {e.name.lower()} (\n  " + ",\n  ".join(columns) + "\n);\n"

        return DatabaseSchema(schema_type="sql", content=content, tables=tables)


    # Génération architecture complète
    async def a_generate_architecture(self, requirements_json: Dict[str, Any],
                                      project_stack: Optional[Dict[str, List[str]]] = None) -> ArchitecturalAnalysis:
        # Détection entités + attributs par LLM
        entities = await self.detect_entities_with_llm(requirements_json)
        uml_diagrams = self.generate_uml(entities, requirements_json)
        dependencies = self.detect_dependencies(entities)
        database = self.generate_database(entities)
        architecture = ArchitectureProposal(pattern="MVC",
                                            description="Architecture basée sur entités détectées par LLM",
                                            diagram=uml_diagrams[1])
        tech_stack = TechStack(
            frontend=project_stack.get("frontend", ["React", "TypeScript"]) if project_stack else ["React", "TypeScript"],
            backend=project_stack.get("backend", ["FastAPI", "Python"]) if project_stack else ["FastAPI", "Python"],
            database=project_stack.get("database", ["PostgreSQL"]) if project_stack else ["PostgreSQL"],
            infrastructure=project_stack.get("infrastructure", ["Docker", "AWS"]) if project_stack else ["Docker", "AWS"],
            justification="Stack technique générée automatiquement"
        )
        consistency = ConsistencyCheck(is_consistent=True, issues=[])
        complexity = ComplexityEstimation(technical_score=7, dev_days_estimate=30, reasoning="Complexité moyenne")
        summary = f"Architecture générée pour {len(entities)} entités principales avec UML, DB et stack technique."

        return ArchitecturalAnalysis(
            summary=summary,
            entities=entities,
            architecture=architecture,
            uml_diagrams=uml_diagrams,
            database=database,
            dependencies=dependencies,
            tech_stack=tech_stack,
            consistency=consistency,
            complexity=complexity,
            raw_model_output=requirements_json
        )

    def generate_architecture(self, requirements_json: Dict[str, Any],
                              project_stack: Optional[Dict[str, List[str]]] = None) -> ArchitecturalAnalysis:
        return asyncio.run(self.a_generate_architecture(requirements_json, project_stack))
