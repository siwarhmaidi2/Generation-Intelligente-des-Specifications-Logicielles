import asyncio
import json
import re
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
from agent_analyst.llm_client import LLMClient, ChatMessage
from .prompts import ARCHITECT_SYSTEM_PROMPT


# =====================================================
# Schemas
# =====================================================

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
    technical_score: float
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


# =====================================================
# Agent Architect
# =====================================================

class AgentArchitect:

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    # -------------------------------------------------
    # JSON Cleaning
    # -------------------------------------------------

    @staticmethod
    def _clean_json(text: str) -> str:
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        text = re.sub(r"\r\n", "\n", text)
        # échappe les backslashes invalides
        text = re.sub(r'\\_', '_', text)
        text = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r"\\\\", text)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        return text.strip()

    


    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = AgentArchitect._clean_json(text)
            return json.loads(cleaned)

    # -------------------------------------------------
    # Entity Detection via LLM
    # -------------------------------------------------

    async def detect_entities_with_llm(self, requirements_json: Dict[str, Any]) -> List[Entity]:

        prompt = f"""
        Analyse les exigences suivantes :

        {json.dumps(requirements_json, ensure_ascii=False, indent=2)}

        Identifie les entités principales avec leurs attributs et relations.

        Retourne uniquement un JSON :
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

        response = await self.llm_client.acomplete(messages, max_tokens=2000)
        content = response["choices"][0]["message"]["content"]

        data = self._parse_json(content)
        return [Entity(**e) for e in data.get("entities", [])]

    # -------------------------------------------------
    # UML Generation
    # -------------------------------------------------

    @staticmethod
    def generate_uml(entities: List[Entity], requirements_json: Dict[str, Any]) -> List[UMLDiagram]:

        class_diagram = "@startuml\n"
        for e in entities:
            class_diagram += f"class {e.name} {{\n"
            for attr in e.attributes:
                class_diagram += f"  {attr}: string\n"
            class_diagram += "}\n"

        for e in entities:
            for rel in e.relations:
                class_diagram += f"{e.name} --> {rel}\n"

        class_diagram += "@enduml"

        use_case = "@startuml\n"
        actors = [a["name"] for a in requirements_json.get("actors", [])]

        for actor in actors:
            alias = actor.replace(" ", "_")
            use_case += f"actor \"{actor}\" as {alias}\n"

        for fr in requirements_json.get("functional_requirements", []):
            use_case += f"usecase \"{fr['description']}\" as {fr['id']}\n"
            for actor in actors:
                use_case += f"{actor.replace(' ', '_')} --> {fr['id']}\n"

        use_case += "@enduml"

        sequence = "@startuml\n"
        if actors and entities and requirements_json.get("functional_requirements"):
            actor = actors[0].replace(" ", "_")
            entity = entities[0].name
            action = requirements_json["functional_requirements"][0]["description"]
            sequence += f"{actor} -> {entity}: {action}\n"
            sequence += f"{entity} --> {actor}: response\n"
        sequence += "@enduml"

        return [
            UMLDiagram(diagram_type="class", content=class_diagram),
            UMLDiagram(diagram_type="use_case", content=use_case),
            UMLDiagram(diagram_type="sequence", content=sequence)
        ]

    # -------------------------------------------------
    # Database Generation (Smart Types + FK)
    # -------------------------------------------------
    @staticmethod
    def generate_database(entities: List[Entity]) -> DatabaseSchema:
        import re

        def to_snake_case(name: str) -> str:
            """Convertit les noms avec espaces en snake_case et supprime les caractères spéciaux."""
            return re.sub(r'\W+', '_', name.strip().lower())

        tables = []
        content = ""
        entity_names = {to_snake_case(e.name) for e in entities}

        for e in entities:
            table_name = to_snake_case(e.name)
            columns = []
            foreign_keys = []
            has_pk = False

            # Colonnes principales
            for attr in e.attributes:
                attr_snake = to_snake_case(attr)
                attr_lower = attr.lower()
                if attr_lower == "id":
                    columns.append("id INT PRIMARY KEY")
                    has_pk = True
                elif "date" in attr_lower or "at" in attr_lower:
                    columns.append(f"{attr_snake} TIMESTAMP")
                elif "status" in attr_lower:
                    # pour les colonnes status → VARCHAR(50)
                    columns.append(f"{attr_snake} VARCHAR(50)")
                elif attr_lower in ["updated_at", "created_at", "created__at", "updated__at"]:
                    # timestamp pour dates
                    columns.append(f"{attr_snake} TIMESTAMP")

                elif "count" in attr_lower or "number" in attr_lower:
                    columns.append(f"{attr_snake} INT")
                else:
                    columns.append(f"{attr_snake} VARCHAR(255)")

            if not has_pk:
                columns.insert(0, "id INT PRIMARY KEY")

            # Relations → Foreign Keys
            for rel in e.relations:
                rel_snake = to_snake_case(rel)
                fk_col = f"{rel_snake}_id"
                if rel_snake in entity_names and fk_col not in [c.split()[0] for c in columns]:
                    columns.append(f"{fk_col} INT")
                    foreign_keys.append(
                        f"FOREIGN KEY ({fk_col}) REFERENCES {rel_snake}(id) ON DELETE CASCADE"
                    )

            table_def = columns + foreign_keys
            tables.append(DatabaseTable(name=table_name, columns=table_def))

            content += f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(table_def) + "\n);\n\n"

        return DatabaseSchema(
            schema_type="sql",
            content=content,
            tables=tables
        )


    # -------------------------------------------------
    # Dependencies
    # -------------------------------------------------

    @staticmethod
    def detect_dependencies(entities: List[Entity]) -> List[ModuleDependency]:
        deps = []
        for e in entities:
            for rel in e.relations:
                deps.append(ModuleDependency(
                    module_from=e.name,
                    module_to=rel,
                    description="Relation détectée"
                ))
        return deps

    # -------------------------------------------------
    # Complexity Estimation
    # -------------------------------------------------

    @staticmethod
    def compute_complexity(requirements_json: Dict[str, Any],
                           entities: List[Entity]) -> ComplexityEstimation:

        nb_entities = len(entities)
        nb_actors = len(requirements_json.get("actors", []))
        nb_fr = len(requirements_json.get("functional_requirements", []))
        nb_nfr = len(requirements_json.get("non_functional_requirements", []))
        nb_rules = len(requirements_json.get("business_rules", []))

        relation_weight = sum(len(e.relations) for e in entities) * 0.8

        raw_score = (
            nb_entities * 1.5 +
            nb_actors * 1.0 +
            nb_fr * 0.7 +
            nb_nfr * 1.8 +
            nb_rules * 0.6 +
            relation_weight
        )

        technical_score = round(min(raw_score / 4, 10), 2)
        dev_days = int(technical_score * 10 + nb_entities * 3)

        reasoning = (
            f"{nb_entities} entités, {nb_actors} acteurs, "
            f"{nb_fr} FR, {nb_nfr} NFR, "
            f"{relation_weight:.1f} poids relationnel."
        )

        return ComplexityEstimation(
            technical_score=technical_score,
            dev_days_estimate=dev_days,
            reasoning=reasoning
        )

    # -------------------------------------------------
    # Dynamic Architecture Choice
    # -------------------------------------------------

    @staticmethod
    def choose_architecture(complexity: ComplexityEstimation) -> ArchitectureProposal:

        if complexity.technical_score > 8:
            pattern = "Microservices"
            description = "Architecture distribuée adaptée aux systèmes évolutifs."
        elif complexity.technical_score > 6:
            pattern = "Modular Monolith"
            description = "Architecture modulaire avec séparation forte des domaines."
        else:
            pattern = "Layered Architecture"
            description = "Architecture en couches adaptée aux projets simples ou moyens."

        return ArchitectureProposal(pattern=pattern, description=description)

    # -------------------------------------------------
    # Dynamic Stack Generation
    # -------------------------------------------------

        # -------------------------------------------------
    # Dynamic Stack Generation via LLM
    # -------------------------------------------------

    async def generate_dynamic_stack_with_llm(
        self,
        requirements_json: Dict[str, Any],
        complexity: ComplexityEstimation
    ) -> TechStack:
        """
        Génère dynamiquement un tech stack via LLM basé sur
        le résumé du projet, les entités, FR/NFR et le score technique.
        """

        prompt = f"""
        Tu es un expert en architecture logicielle.
        Projet résumé : {requirements_json.get('summary', 'Projet logiciel')}
        Entités : {[e['name'] for e in requirements_json.get('entities', [])]}
        Fonctionnalités : {[fr['description'] for fr in requirements_json.get('functional_requirements', [])]}
        Non-fonctionnelles : {[nfr['description'] for nfr in requirements_json.get('non_functional_requirements', [])]}
        Score technique : {complexity.technical_score}

        Propose un stack complet : frontend, backend, database, infrastructure,
        adapté au projet. Justifie chaque choix.
        Réponds strictement en JSON comme ceci :
        {{
            "frontend": [...],
            "backend": [...],
            "database": [...],
            "infrastructure": [...],
            "justification": "..."
        }}
        """

        messages = [
            ChatMessage(role="system", content=ARCHITECT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt)
        ]

        try:
            response = await self.llm_client.acomplete(messages, max_tokens=1500)
            content = response["choices"][0]["message"]["content"]
            stack_data = self._parse_json(content)
            return TechStack(**stack_data)
        except Exception as e:
            # fallback simple basé sur score
            if complexity.technical_score > 8:
                infra = ["Containerized Deployment", "Cloud Infrastructure"]
            elif complexity.technical_score > 6:
                infra = ["Containerized Deployment"]
            else:
                infra = ["Single Server Deployment"]

            return TechStack(
                frontend=["Modern Web Client"],
                backend=["RESTful Backend Service"],
                database=["Relational Database"],
                infrastructure=infra,
                justification=f"Fallback stack basée sur score {complexity.technical_score}"
            )


    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    @staticmethod
    def generate_summary(requirements_json: Dict[str, Any],
                         entities: List[Entity],
                         complexity: ComplexityEstimation) -> str:

        base_summary = requirements_json.get("summary", "Projet logiciel")

        return (
            f"Architecture conçue pour {base_summary}. "
            f"{len(entities)} entités principales identifiées. "
            f"Score technique estimé: {complexity.technical_score}/10. "
            f"Charge estimée: {complexity.dev_days_estimate} jours."
        )

    # -------------------------------------------------
    # Main Generation
    # -------------------------------------------------

    async def a_generate_architecture(
        self,
        requirements_json: Dict[str, Any],
        project_stack: Optional[Dict[str, List[str]]] = None
    ) -> ArchitecturalAnalysis:

        entities = await self.detect_entities_with_llm(requirements_json)

        complexity = self.compute_complexity(requirements_json, entities)

        architecture = self.choose_architecture(complexity)

        tech_stack = (
            TechStack(**project_stack)
            if project_stack
            else await self.generate_dynamic_stack_with_llm(requirements_json, complexity)
        )


        uml_diagrams = self.generate_uml(entities, requirements_json)
        database = self.generate_database(entities)
        dependencies = self.detect_dependencies(entities)

        summary = self.generate_summary(requirements_json, entities, complexity)

        consistency = ConsistencyCheck(
            is_consistent=len(entities) > 0,
            issues=[] if entities else ["Aucune entité détectée"]
        )

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

    def generate_architecture(
        self,
        requirements_json: Dict[str, Any],
        project_stack: Optional[Dict[str, List[str]]] = None
    ) -> ArchitecturalAnalysis:

        return asyncio.run(
            self.a_generate_architecture(requirements_json, project_stack)
        )
