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
    relations: List[Dict[str, str]] = Field(default_factory=list)

class TechStack(BaseModel):
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)
    justification: str = ""

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
    uml_diagrams: List[UMLDiagram]
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

    # --------------------------
    # JSON Cleaning
    # --------------------------

    @staticmethod
    def _clean_json(text: str) -> str:
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        text = re.sub(r"\r\n", "\n", text)
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

    # --------------------------
    # Entity Detection via LLM
    # --------------------------

    async def detect_entities_with_llm(self, requirements_json: Dict[str, Any]) -> List[Entity]:
        """Détecte les entités avec un format structuré strict"""
        
        project_summary = requirements_json.get("summary", "")
        functional_reqs = requirements_json.get("functional_requirements", [])
        actors = requirements_json.get("actors", [])
        
        context = f"""
CONTEXTE DU PROJET : {project_summary}

ACTEURS DU SYSTÈME :
{json.dumps(actors, ensure_ascii=False, indent=2)}

EXIGENCES FONCTIONNELLES :
{json.dumps(functional_reqs, ensure_ascii=False, indent=2)}
"""
        
        prompt = f"""
Tu es un architecte logiciel expert. Analyse les exigences suivantes et identifie UNIQUEMENT les entités métier pertinentes pour CE projet spécifique.

{context}

IMPORTANT : 
- Analyse le CONTEXTE RÉEL du projet fourni ci-dessus
- Identifie les entités MÉTIER mentionnées ou sous-entendues dans les exigences
- Base-toi UNIQUEMENT sur les informations fournies
- N'invente AUCUNE entité générique ou exemple
- ADAPTE les entités au domaine métier décrit dans le contexte

Retourne UNIQUEMENT un JSON valide avec ce format EXACT :

{{
    "entities": [
        {{
            "name": "NomEntite",
            "attributes": [
                "id:int",
                "attribute1:string",
                "attribute2:datetime",
                "attribute3:boolean"
            ],
            "relations": [
                {{"type": "inherits", "target": "ParentEntity"}},
                {{"type": "has_many", "target": "OtherEntity"}},
                {{"type": "belongs_to", "target": "MainEntity"}}
            ]
        }}
    ]
}}

RÈGLES STRICTES :
1. Les noms d'entités doivent être en ANGLAIS et refléter le métier réel décrit
2. Attributs au format "nom:type" (types: string, int, float, boolean, datetime, date, text)
3. Relations avec structure obligatoire {{"type": "...", "target": "..."}}
4. Types de relations autorisés : inherits, has_many, has_one, belongs_to, many_to_many
5. N'utilise AUCUN exemple prédéfini - base-toi uniquement sur le contexte fourni
6. Concentre-toi sur les entités MÉTIER décrites dans les exigences

Analyse maintenant les exigences fournies et retourne les entités ADAPTÉES au contexte réel.
Retourne UNIQUEMENT le JSON, sans texte avant ou après.
"""

        messages = [
            ChatMessage(role="system", content=ARCHITECT_SYSTEM_PROMPT + "\n\nTu dois analyser le contexte métier réel fourni et proposer des entités cohérentes avec le domaine applicatif décrit."),
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=3500)
        content = response["choices"][0]["message"]["content"]
        data = self._parse_json(content)
        
        entities = []
        for e_data in data.get("entities", []):
            relations = []
            for rel in e_data.get("relations", []):
                if isinstance(rel, dict):
                    relations.append(rel)
                elif isinstance(rel, str):
                    relations.append(self._parse_legacy_relation(rel))
            
            e_data["relations"] = relations
            entities.append(Entity(**e_data))
        
        return entities

    @staticmethod
    def _parse_legacy_relation(rel_str: str) -> Dict[str, str]:
        """Convertit les anciennes relations string en format structuré"""
        rel_lower = rel_str.lower().strip()
        
        if any(keyword in rel_lower for keyword in ["inherits", "extends", "hérite"]):
            match = re.search(r"(?:inherits?|extends?|hérite\s+de)\s*:?\s*(\w+)", rel_lower)
            if match:
                return {"type": "inherits", "target": match.group(1)}
        
        if ":" in rel_str:
            parts = rel_str.split(":", 1)
            return {"type": parts[0].strip(), "target": parts[1].strip()}
        
        return {"type": "has_many", "target": rel_str.strip()}

    # --------------------------
    # UML Generation
    # --------------------------

    async def generate_uml_diagrams(self, entities: List[Entity], requirements_json: Dict[str, Any]) -> List[UMLDiagram]:
        """Génère les diagrammes UML via LLM pour cohérence maximale"""
        
        entities_data = [
            {
                "name": e.name,
                "attributes": e.attributes,
                "relations": e.relations
            }
            for e in entities
        ]
        
        diagrams = []
        
        # 1. Diagramme de classes
        class_diagram = await self._generate_class_diagram_llm(entities_data, requirements_json)
        diagrams.append(UMLDiagram(
            diagram_type="class",
            content=class_diagram,
            description="Diagramme de classes du système"
        ))
        
        # 2. Diagramme de cas d'utilisation
        usecase_diagram = await self._generate_usecase_diagram_llm(requirements_json, entities_data)
        diagrams.append(UMLDiagram(
            diagram_type="use_case",
            content=usecase_diagram,
            description="Diagramme des cas d'utilisation"
        ))
        
        # 3. Diagramme d'activité
        activity_diagram = await self._generate_activity_diagram_llm(requirements_json, entities_data)
        diagrams.append(UMLDiagram(
            diagram_type="activity",
            content=activity_diagram,
            description="Diagramme d'activité principal"
        ))
        
        # 4. Diagramme de séquence
        sequence_diagram = await self._generate_sequence_diagram_llm(requirements_json, entities_data)
        if sequence_diagram:
            diagrams.append(UMLDiagram(
                diagram_type="sequence",
                content=sequence_diagram,
                description="Diagramme de séquence des flux principaux"
            ))
        
        return diagrams

    async def _generate_class_diagram_llm(self, entities_data: List[Dict], requirements_json: Dict[str, Any]) -> str:
        """Génère le diagramme de classes via LLM"""
        
        project_context = requirements_json.get("summary", "")
        
        prompt = f"""
Génère un diagramme de classes PlantUML VALIDE et COHÉRENT pour ce projet :

CONTEXTE DU PROJET : {project_context}

ENTITÉS DU SYSTÈME :
{json.dumps(entities_data, ensure_ascii=False, indent=2)}

RÈGLES OBLIGATOIRES :
1. Format PlantUML strict : @startuml ... @enduml
2. Utilise EXACTEMENT les entités fournies (ne modifie pas les noms)
3. Respecte les types d'attributs fournis
4. Traduis correctement les relations :
   - inherits → ParentClass <|-- ChildClass
   - has_many → ClassA "1" -- "0..*" ClassB
   - has_one → ClassA "1" -- "0..1" ClassB
   - belongs_to → ClassB "1" -- "0..*" ClassA
   - many_to_many → ClassA "*" -- "*" ClassB
5. Ajoute des labels pertinents sur les relations basés sur le contexte métier
6. Évite les doublons de relations
7. Ajoute des notes si cela clarifie l'architecture

FORMAT ATTENDU :
@startuml
class EntityName1 {{
  attribute1: type1
  attribute2: type2
}}

class EntityName2 {{
  attribute1: type1
  attribute2: type2
}}

EntityName1 "1" -- "0..*" EntityName2 : relationship_label >

note right of EntityName1
  Description pertinente
end note

@enduml

Génère maintenant le diagramme COMPLET en utilisant TOUTES les entités fournies.
Base-toi sur le contexte métier pour nommer les relations de manière pertinente.
Retourne UNIQUEMENT le code PlantUML, sans texte explicatif.
"""

        messages = [
            ChatMessage(role="system", content="Tu es un expert UML. Tu génères du code PlantUML valide en respectant EXACTEMENT les entités fournies."),
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=2500)
        content = response["choices"][0]["message"]["content"]
        
        uml_match = re.search(r'@startuml.*?@enduml', content, re.DOTALL)
        if uml_match:
            return uml_match.group(0)
        
        return self._generate_class_diagram_fallback(entities_data)

    async def _generate_usecase_diagram_llm(self, requirements_json: Dict[str, Any], entities_data: List[Dict]) -> str:
        """Génère le diagramme de cas d'utilisation via LLM"""
        
        project_context = requirements_json.get("summary", "")
        actors = requirements_json.get("actors", [])
        functional_reqs = requirements_json.get("functional_requirements", [])
        
        prompt = f"""
Génère un diagramme de cas d'utilisation PlantUML VALIDE pour ce projet :

CONTEXTE DU PROJET : {project_context}

ACTEURS DU SYSTÈME :
{json.dumps(actors, ensure_ascii=False, indent=2)}

EXIGENCES FONCTIONNELLES :
{json.dumps(functional_reqs, ensure_ascii=False, indent=2)}

ENTITÉS MÉTIER (pour contexte) :
{json.dumps([e['name'] for e in entities_data], ensure_ascii=False)}

RÈGLES OBLIGATOIRES :
1. Format PlantUML strict : @startuml ... @enduml
2. Déclare tous les acteurs fournis : actor "Nom Complet" as alias
3. Crée des use cases basés UNIQUEMENT sur les exigences fonctionnelles fournies
4. Regroupe par package selon les thématiques identifiées dans les exigences
5. Relie acteurs aux UC de manière logique selon leurs rôles décrits
6. Utilise <<include>> pour les dépendances obligatoires identifiées
7. Utilise <<extend>> pour les extensions optionnelles identifiées
8. Garde la cohérence avec le contexte métier fourni

FORMAT ATTENDU :
@startuml
left to right direction

actor "ActorName1" as actor1
actor "ActorName2" as actor2

package "Theme1" {{
  usecase "Use case description" as UC1
  usecase "Use case description" as UC2
}}

package "Theme2" {{
  usecase "Use case description" as UC3
}}

actor1 --> UC1
actor2 --> UC2
actor2 --> UC3

UC1 ..> UC2 : <<include>>

@enduml

Génère maintenant le diagramme basé UNIQUEMENT sur les données fournies.
Identifie les thèmes dans les exigences fonctionnelles pour grouper les use cases.
Retourne UNIQUEMENT le code PlantUML.
"""

        messages = [
            ChatMessage(role="system", content="Tu es un expert UML. Tu génères des use cases cohérents avec le contexte métier fourni."),
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=2500)
        content = response["choices"][0]["message"]["content"]
        
        uml_match = re.search(r'@startuml.*?@enduml', content, re.DOTALL)
        if uml_match:
            return uml_match.group(0)
        
        return self._generate_usecase_diagram_fallback(requirements_json)

    async def _generate_activity_diagram_llm(self, requirements_json: Dict[str, Any], entities_data: List[Dict]) -> str:
        """Génère un diagramme d'activité via LLM"""
        
        project_context = requirements_json.get("summary", "")
        functional_reqs = requirements_json.get("functional_requirements", [])
        
        prompt = f"""
Génère un diagramme d'activité PlantUML représentant le WORKFLOW PRINCIPAL du système :

CONTEXTE DU PROJET : {project_context}

EXIGENCES FONCTIONNELLES :
{json.dumps(functional_reqs, ensure_ascii=False, indent=2)}

ENTITÉS MÉTIER :
{json.dumps([e['name'] for e in entities_data], ensure_ascii=False)}

RÈGLES OBLIGATOIRES :
1. Format PlantUML strict : @startuml ... @enduml
2. Commence par start, termine par stop
3. Inclus des décisions (if/else) pertinentes basées sur les exigences
4. Utilise fork/join pour les actions parallèles si identifiées dans les exigences
5. Ajoute des partitions (swimlanes) si plusieurs acteurs impliqués
6. Modélise un workflow RÉALISTE basé sur les exigences fournies
7. N'invente aucune étape non décrite dans les exigences

FORMAT ATTENDU :
@startuml
|Actor1|
start

:Activity from requirements;

:Another activity;

if (Condition based on requirements ?) then (yes)
  :Action if true;
  
  |Actor2|
  :Another actor's action;
  
else (no)
  :Action if false;
  stop
endif

|Actor1|
:Final activity;

stop
@enduml

Génère maintenant le diagramme basé sur le workflow décrit dans les exigences.
Retourne UNIQUEMENT le code PlantUML.
"""

        messages = [
            ChatMessage(role="system", content="Tu es un expert UML. Tu modélises des workflows réalistes basés uniquement sur les exigences fournies."),
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=3000)
        content = response["choices"][0]["message"]["content"]
        
        uml_match = re.search(r'@startuml.*?@enduml', content, re.DOTALL)
        if uml_match:
            return uml_match.group(0)
        
        return self._generate_activity_diagram_fallback(requirements_json)

    async def _generate_sequence_diagram_llm(self, requirements_json: Dict[str, Any], entities_data: List[Dict]) -> Optional[str]:
        """Génère un diagramme de séquence pour les flux critiques"""
        
        functional_reqs = requirements_json.get("functional_requirements", [])
        if not functional_reqs:
            return None
        
        # Prendre la première exigence comme flux principal
        main_flow = functional_reqs[0] if functional_reqs else {}
        
        prompt = f"""
Génère un diagramme de séquence PlantUML pour ce flux :

FLUX À MODÉLISER :
{json.dumps(main_flow, ensure_ascii=False, indent=2)}

ENTITÉS DU SYSTÈME :
{json.dumps([e['name'] for e in entities_data], ensure_ascii=False)}

CONTEXTE : {requirements_json.get("summary", "")}

RÈGLES OBLIGATOIRES :
1. Format PlantUML strict : @startuml ... @enduml
2. Déclare les participants (acteur + entités métier pertinentes)
3. Messages avec flèches appropriées :
   - -> : appel synchrone
   - --> : retour
   - ->> : appel asynchrone
4. Utilise activate/deactivate pour les lifelines
5. Inclus alt/else pour les scénarios alternatifs si pertinent
6. Base-toi uniquement sur le flux fourni
7. N'invente aucune interaction non décrite

FORMAT ATTENDU :
@startuml
actor "ActorName" as actor
participant "EntityName1" as entity1
participant "EntityName2" as entity2

actor -> entity1: action()
activate entity1

entity1 -> entity2: relatedAction()
activate entity2
entity2 --> entity1: result
deactivate entity2

alt condition
  entity1 --> actor: success
else
  entity1 --> actor: error
end

deactivate entity1

@enduml

Génère maintenant le diagramme pour le flux fourni.
Retourne UNIQUEMENT le code PlantUML.
"""

        messages = [
            ChatMessage(role="system", content="Tu es un expert UML. Tu modélises des séquences d'interactions basées uniquement sur les informations fournies."),
            ChatMessage(role="user", content=prompt)
        ]

        response = await self.llm_client.acomplete(messages, max_tokens=2500)
        content = response["choices"][0]["message"]["content"]
        
        uml_match = re.search(r'@startuml.*?@enduml', content, re.DOTALL)
        if uml_match:
            return uml_match.group(0)
        
        return None

    # --------------------------
    # Fallback Methods
    # --------------------------

    def _generate_class_diagram_fallback(self, entities_data: List[Dict]) -> str:
        """Génération de secours du diagramme de classes"""
        diagram = "@startuml\n\n"
        
        entity_names = {e['name'] for e in entities_data}
        added_relations = set()
        
        for e in entities_data:
            diagram += f"class {e['name']} {{\n"
            for attr in e.get('attributes', []):
                if ':' in attr:
                    name, type_attr = attr.split(':', 1)
                    diagram += f"  {name}: {type_attr}\n"
                else:
                    diagram += f"  {attr}: string\n"
            diagram += "}\n\n"
        
        for e in entities_data:
            for rel in e.get('relations', []):
                if isinstance(rel, dict):
                    rel_type = rel.get('type', 'has_many')
                    target = rel.get('target', '')
                    
                    if target not in entity_names:
                        continue
                    
                    source = e['name']
                    
                    if rel_type == 'inherits':
                        diagram += f"{target} <|-- {source}\n"
                    elif rel_type == 'has_many':
                        rel_key = tuple(sorted([source, target]))
                        if rel_key not in added_relations:
                            diagram += f"{source} \"1\" -- \"0..*\" {target}\n"
                            added_relations.add(rel_key)
                    elif rel_type == 'has_one':
                        rel_key = tuple(sorted([source, target]))
                        if rel_key not in added_relations:
                            diagram += f"{source} \"1\" -- \"0..1\" {target}\n"
                            added_relations.add(rel_key)
                    elif rel_type == 'belongs_to':
                        rel_key = tuple(sorted([source, target]))
                        if rel_key not in added_relations:
                            diagram += f"{target} \"1\" -- \"0..*\" {source}\n"
                            added_relations.add(rel_key)
                    elif rel_type == 'many_to_many':
                        rel_key = tuple(sorted([source, target]))
                        if rel_key not in added_relations:
                            diagram += f"{source} \"*\" -- \"*\" {target}\n"
                            added_relations.add(rel_key)
        
        diagram += "\n@enduml"
        return diagram

    def _generate_usecase_diagram_fallback(self, requirements_json: Dict[str, Any]) -> str:
        """Génération de secours du diagramme de cas d'utilisation"""
        diagram = "@startuml\nleft to right direction\n\n"
        
        actors = requirements_json.get("actors", [])
        functional_reqs = requirements_json.get("functional_requirements", [])
        
        for actor in actors:
            name = actor.get("name", "Actor")
            alias = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            diagram += f"actor \"{name}\" as {alias}\n"
        
        diagram += "\n"
        
        # Grouper par thème détecté
        grouped_uc = {}
        for i, fr in enumerate(functional_reqs):
            desc = fr.get("description", "Use case")
            uc_id = fr.get("id", f"UC{i+1}")
            
            # Essayer de détecter un thème dans la description
            theme = "Fonctionnalités principales"
            desc_lower = desc.lower()
            
            # Chercher des mots-clés génériques de groupement
            for keyword in ["gestion", "administration", "rapport", "recherche", "validation", "migration"]:
                if keyword in desc_lower:
                    theme = keyword.capitalize()
                    break
            
            if theme not in grouped_uc:
                grouped_uc[theme] = []
            grouped_uc[theme].append((uc_id, desc[:60]))
        
        for theme, ucs in grouped_uc.items():
            diagram += f"\npackage \"{theme}\" {{\n"
            for uc_id, desc in ucs:
                diagram += f"  usecase \"{desc}\" as {uc_id}\n"
            diagram += "}\n"
        
        diagram += "\n"
        
        # Associations basées sur les rôles et responsabilités
        for fr in functional_reqs:
            uc_id = fr.get("id", "")
            desc = fr.get("description", "").lower()
            
            for actor in actors:
                actor_name = actor.get("name", "")
                alias = re.sub(r'[^a-zA-Z0-9_]', '_', actor_name)
                role = actor.get("role", "").lower()
                responsibilities = [r.lower() for r in actor.get("responsibilities", [])]
                
                # Match si le nom de l'acteur, son rôle ou ses responsabilités apparaissent
                if (actor_name.lower() in desc or 
                    role in desc or 
                    any(resp in desc for resp in responsibilities)):
                    diagram += f"{alias} --> {uc_id}\n"
        
        diagram += "\n@enduml"
        return diagram

    def _generate_activity_diagram_fallback(self, requirements_json: Dict[str, Any]) -> str:
        """Génération de secours du diagramme d'activité"""
        diagram = "@startuml\nstart\n\n"
        
        functional_reqs = requirements_json.get("functional_requirements", [])[:6]
        
        if functional_reqs:
            first_req = functional_reqs[0].get("description", "Initialisation")
            diagram += f":{first_req};\n\n"
            
            if len(functional_reqs) > 1:
                diagram += "if (Action réussie ?) then (oui)\n"
                for req in functional_reqs[1:]:
                    desc = req.get("description", "Action")
                    diagram += f"  :{desc};\n"
                diagram += "else (non)\n"
                diagram += "  :Afficher erreur;\n"
                diagram += "  stop\n"
                diagram += "endif\n"
        
        diagram += "\nstop\n@enduml"
        return diagram

    # --------------------------
    # Complexity & Tech Stack
    # --------------------------

    @staticmethod
    def compute_complexity(requirements_json: Dict[str, Any], entities: List[Entity]) -> ComplexityEstimation:
        nb_entities = len(entities)
        nb_actors = len(requirements_json.get("actors", []))
        nb_fr = len(requirements_json.get("functional_requirements", []))
        nb_nfr = len(requirements_json.get("non_functional_requirements", []))
        total_relations = sum(len(e.relations) for e in entities)
        relation_weight = total_relations * 0.8
        
        raw_score = nb_entities * 1.5 + nb_actors * 1.0 + nb_fr * 0.7 + nb_nfr * 1.8 + relation_weight
        technical_score = round(min(raw_score / 4, 10), 2)
        dev_days = int(technical_score * 10 + nb_entities * 3)
        
        reasoning = (
            f"{nb_entities} entités, {nb_actors} acteurs, {nb_fr} exigences fonctionnelles, "
            f"{nb_nfr} exigences non-fonctionnelles, {total_relations} relations."
        )
        
        return ComplexityEstimation(
            technical_score=technical_score,
            dev_days_estimate=dev_days,
            reasoning=reasoning
        )

    async def generate_dynamic_stack_with_llm(self, requirements_json: Dict[str, Any], complexity: ComplexityEstimation) -> TechStack:
        """Génère une stack technique adaptée"""
        
        nfr = requirements_json.get('non_functional_requirements', [])
        project_summary = requirements_json.get('summary', '')
        
        prompt = f"""
Recommande une stack technique MODERNE et ADAPTÉE pour ce projet :

CONTEXTE DU PROJET : {project_summary}

SCORE DE COMPLEXITÉ : {complexity.technical_score}/10

EXIGENCES NON-FONCTIONNELLES :
{json.dumps(nfr, ensure_ascii=False, indent=2)}

Retourne UNIQUEMENT un JSON :
{{
    "frontend": ["Technology1", "Technology2", "Technology3"],
    "backend": ["Technology1", "Technology2", "Technology3", "Technology4"],
    "infrastructure": ["Technology1", "Technology2", "Technology3"],
    "justification": "Explication claire en 2-3 phrases"
}}

RÈGLES :
- Complexité < 4 : Stack simple et légère
- Complexité 4-7 : Stack moderne et équilibrée
- Complexité > 7 : Stack enterprise et scalable
- Considère les exigences non-fonctionnelles fournies
- Sois spécifique (noms de frameworks/technologies précis)
- Justifie tes choix par rapport au contexte et aux contraintes

Retourne le JSON uniquement.
"""

        messages = [
            ChatMessage(role="system", content="Tu es un architecte logiciel. Tu recommandes des stacks techniques modernes et justifiées."),
            ChatMessage(role="user", content=prompt)
        ]

        try:
            response = await self.llm_client.acomplete(messages, max_tokens=1500)
            content = response["choices"][0]["message"]["content"]
            data = self._parse_json(content)
            
            return TechStack(
                frontend=data.get("frontend", ["React", "TypeScript"]),
                backend=data.get("backend", ["Node.js", "Express", "PostgreSQL"]),
                infrastructure=data.get("infrastructure", ["Docker", "Cloud"]),
                justification=data.get("justification", f"Stack adaptée à un score de {complexity.technical_score}/10")
            )
        except Exception:
            # Fallback intelligent basé uniquement sur le score
            if complexity.technical_score > 7:
                return TechStack(
                    frontend=["React", "TypeScript", "Material-UI", "Redux Toolkit"],
                    backend=["Node.js", "NestJS", "PostgreSQL", "Redis", "Elasticsearch"],
                    infrastructure=["Docker", "Kubernetes", "Cloud Platform"],
                    justification=f"Architecture enterprise pour haute complexité ({complexity.technical_score}/10) avec focus sur scalabilité et sécurité"
                )
            elif complexity.technical_score > 4:
                return TechStack(
                    frontend=["React", "TypeScript", "Tailwind CSS"],
                    backend=["Node.js", "Express", "PostgreSQL", "Redis"],
                    infrastructure=["Docker", "Cloud Platform"],
                    justification=f"Stack moderne et équilibrée pour complexité moyenne ({complexity.technical_score}/10)"
                )
            else:
                return TechStack(
                    frontend=["Vue.js", "JavaScript", "Bootstrap"],
                    backend=["Python", "FastAPI", "SQLite"],
                    infrastructure=["Docker", "VPS"],
                    justification=f"Stack simple et efficace pour faible complexité ({complexity.technical_score}/10)"
                )

    @staticmethod
    def generate_summary(requirements_json: Dict[str, Any], entities: List[Entity], complexity: ComplexityEstimation) -> str:
        base_summary = requirements_json.get("summary", "Projet logiciel")
        entity_names = ", ".join([e.name for e in entities[:5]])
        suffix = "..." if len(entities) > 5 else ""
        
        return (
            f"Architecture conçue pour : {base_summary}. "
            f"Le système comprend {len(entities)} entités principales ({entity_names}{suffix}). "
            f"Score de complexité technique : {complexity.technical_score}/10. "
            f"Charge de travail estimée : {complexity.dev_days_estimate} jours de développement."
        )

    def validate_architecture(self, entities: List[Entity], requirements_json: Dict[str, Any]) -> ConsistencyCheck:
        """Valide la cohérence de l'architecture"""
        issues = []
        
        if not entities:
            issues.append("Aucune entité détectée dans le système")
            return ConsistencyCheck(is_consistent=False, issues=issues)
        
        entity_names = {e.name for e in entities}
        
        # Vérifier les relations invalides
        for entity in entities:
            for rel in entity.relations:
                if isinstance(rel, dict):
                    target = rel.get('target', '')
                    if target and target not in entity_names:
                        issues.append(f"Relation invalide : {entity.name} -> {target} (entité cible introuvable)")
        
        # Vérifier la couverture des exigences
        functional_reqs = requirements_json.get("functional_requirements", [])
        if functional_reqs and len(entities) < max(2, len(functional_reqs) / 5):
            issues.append("Nombre d'entités potentiellement insuffisant par rapport aux exigences fonctionnelles")
        
        # Vérifier les acteurs
        actors = requirements_json.get("actors", [])
        if not actors:
            issues.append("Aucun acteur défini dans le système")
        
        is_consistent = len(issues) == 0
        
        return ConsistencyCheck(is_consistent=is_consistent, issues=issues)

    # --------------------------
    # Main Generation
    # --------------------------

    async def a_generate_architecture(self, requirements_json: Dict[str, Any]) -> ArchitecturalAnalysis:
        """Génération complète de l'architecture"""
        
        # 1. Détection des entités métier
        entities = await self.detect_entities_with_llm(requirements_json)
        
        # 2. Calcul de la complexité
        complexity = self.compute_complexity(requirements_json, entities)
        
        # 3. Génération de la stack technique
        tech_stack = await self.generate_dynamic_stack_with_llm(requirements_json, complexity)
        
        # 4. Génération des diagrammes UML cohérents
        uml_diagrams = await self.generate_uml_diagrams(entities, requirements_json)
        
        # 5. Génération du résumé
        summary = self.generate_summary(requirements_json, entities, complexity)
        
        # 6. Validation
        consistency = self.validate_architecture(entities, requirements_json)
        
        return ArchitecturalAnalysis(
            summary=summary,
            entities=entities,
            uml_diagrams=uml_diagrams,
            tech_stack=tech_stack,
            consistency=consistency,
            complexity=complexity,
            raw_model_output=requirements_json
        )

    def generate_architecture(self, requirements_json: Dict[str, Any]) -> ArchitecturalAnalysis:
        """Version synchrone"""
        return asyncio.run(self.a_generate_architecture(requirements_json))