from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, BeforeValidator, model_validator
from typing_extensions import Annotated


class UMLDiagram(BaseModel):
    diagram_type: str = Field(..., description="Type de diagramme (use_case, class, sequence)")
    content: str = Field(..., description="Code PlantUML du diagramme")
    description: Optional[str] = Field(None, description="Description courte")

    @field_validator('diagram_type', mode='before')
    @classmethod
    def normalize_type(cls, v):
        if hasattr(v, 'lower'):
            v = v.lower()
        mapping = {
            "usecasediagram": "use_case",
            "classdiagram": "class",
            "sequencediagram": "sequence"
        }
        return mapping.get(v, v)

class Entity(BaseModel):
    name: str = Field(..., description="Nom de l'entité")
    attributes: List[str] = Field(default_factory=list, description="Liste des attributs")
    relations: List[str] = Field(default_factory=list, description="Relations")

    @field_validator('attributes', mode='before')
    @classmethod
    def parse_attributes(cls, v):
        # Convertir [{'name': 'id', ...}] en ['id', ...]
        if isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    # Essayer de récupérer name ou label
                    new_list.append(item.get('name') or item.get('label') or str(item))
                else:
                    new_list.append(str(item))
            return new_list
        return v

class DatabaseTable(BaseModel):
    name: str = Field(..., description="Nom de la table")
    columns: List[str] = Field(..., description="Liste des colonnes")

class DatabaseSchema(BaseModel):
    schema_type: str = Field(..., description="Format (sql, plantuml_erd)")
    content: str = Field(..., description="Contenu SQL/PlantUML")
    tables: List[DatabaseTable] = Field(default_factory=list)

    @field_validator('schema_type', mode='before')
    @classmethod
    def normalize_type(cls, v):
        if hasattr(v, 'lower'):
            return v.lower()
        return v
    
    @field_validator('tables', mode='before')
    @classmethod
    def validate_tables(cls, v):
        if not isinstance(v, list):
            return []
        return v

class TechStack(BaseModel):
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    database: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)
    justification: str = Field(..., description="Justification")

    @field_validator('frontend', 'backend', 'database', 'infrastructure', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(',')]
        return v

class ArchitectureProposal(BaseModel):
    pattern: str = Field(..., description="Pattern (mvc, microservices, etc.)")
    description: str = Field(..., description="Description")
    diagram: Optional[UMLDiagram] = Field(None)

    @field_validator('pattern', mode='before')
    @classmethod
    def normalize_pattern(cls, v):
        if hasattr(v, 'lower'):
            return v.lower()
        return v

class ModuleDependency(BaseModel):
    module_from: str = Field(alias="module_from")
    module_to: str = Field(alias="module_to")
    description: Optional[str] = None

class ConsistencyCheck(BaseModel):
    is_consistent: bool = Field(..., description="True/False")
    issues: List[str] = Field(default_factory=list)

class ComplexityEstimation(BaseModel):
    technical_score: int = Field(..., description="1-10")
    dev_days_estimate: int = Field(..., description="Jours")
    reasoning: str = Field(..., description="Raison")

    @field_validator('technical_score', 'dev_days_estimate', mode='before')
    @classmethod
    def parse_int(cls, v):
        if isinstance(v, str):
            import re
            nums = re.findall(r'\d+', v)
            if nums:
                return int(nums[0])
            mapping = {"faible": 3, "low": 3, "moyen": 5, "medium": 5, "fort": 8, "high": 8, "élevé": 8}
            val = v.lower()
            for k, score in mapping.items():
                if k in val:
                    return score
            return 5 
        return v

class ArchitecturalAnalysis(BaseModel):
    summary: str = Field(..., description="Résumé")
    entities: List[Entity] = Field(default_factory=list)
    architecture: ArchitectureProposal = Field(...)
    uml_diagrams: List[UMLDiagram] = Field(default_factory=list)
    database: DatabaseSchema = Field(...)
    dependencies: List[ModuleDependency] = Field(default_factory=list)
    tech_stack: TechStack = Field(...)
    consistency: ConsistencyCheck = Field(...)
    complexity: ComplexityEstimation = Field(...)
    raw_model_output: Optional[dict] = Field(None)

    @model_validator(mode='before')
    @classmethod
    def pre_process_data(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
            
        # Correction uml_diagrams
        if 'uml_diagrams' in data and isinstance(data['uml_diagrams'], list):
            for d in data['uml_diagrams']:
                if isinstance(d, dict) and 'type' in d and 'diagram_type' not in d:
                    d['diagram_type'] = d.pop('type')
        
        # Correction database
        if 'database' in data and isinstance(data['database'], dict):
            if 'type' in data['database'] and 'schema_type' not in data['database']:
                data['database']['schema_type'] = data['database'].pop('type')
        
        # Correction dependencies
        if 'dependencies' in data and isinstance(data['dependencies'], list):
            new_deps = []
            for d in data['dependencies']:
                if not isinstance(d, dict):
                    continue
                if 'module_from' in d and 'module_to' in d:
                    new_deps.append(d)
                elif 'module' in d:
                    new_deps.append({
                        'module_from': d['module'], 
                        'module_to': 'Unknown', 
                        'description': d.get('description', '')
                    })
                elif 'source' in d and 'target' in d:
                     new_deps.append({
                        'module_from': d['source'],
                        'module_to': d['target'],
                        'description': d.get('description', '')
                     })
            data['dependencies'] = new_deps

        # Correction consistency
        if 'consistency' in data and isinstance(data['consistency'], dict):
            if 'is_consistent' not in data['consistency']:
                issues = data['consistency'].get('issues') or data['consistency'].get('risques') or []
                data['consistency']['is_consistent'] = (len(issues) == 0)
                if 'issues' not in data['consistency']:
                    data['consistency']['issues'] = issues

        # Correction complexity
        if 'complexity' in data and isinstance(data['complexity'], dict):
            if 'technical_score' not in data['complexity']:
                data['complexity']['technical_score'] = data['complexity'].get('technical', 5)
            if 'dev_days_estimate' not in data['complexity']:
                days = data['complexity'].get('charge_de_travail') or data['complexity'].get('jours', 20)
                data['complexity']['dev_days_estimate'] = days

        return data