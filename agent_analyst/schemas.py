from typing import List, Optional, Literal, Dict

from pydantic import BaseModel, Field


PriorityLevel = Literal["must", "should", "could"]


class Requirement(BaseModel):
    id: str = Field(..., description="Identifiant unique ou logique du besoin")
    description: str = Field(..., description="Texte du besoin")
    type: Literal["fonctionnel", "non_fonctionnel"] = Field(
        ..., description="Type de besoin"
    )
    priority: Optional[PriorityLevel] = Field(
        None, description="Priorité must/should/could si applicable"
    )


class Actor(BaseModel):
    name: str = Field(..., description="Nom de l’acteur (rôle, système, utilisateur)")
    description: Optional[str] = Field(
        None, description="Description courte de l’acteur et de sa responsabilité"
    )


class BusinessRule(BaseModel):
    id: str = Field(..., description="Identifiant de la règle métier")
    description: str = Field(..., description="Texte de la règle métier")
    explicit: bool = Field(
        ..., description="True si la règle est explicite dans le texte, False si implicite"
    )


class RequirementsAnalysis(BaseModel):
    functional_requirements: List[Requirement] = Field(default_factory=list)
    non_functional_requirements: List[Requirement] = Field(default_factory=list)
    actors: List[Actor] = Field(default_factory=list)
    business_rules: List[BusinessRule] = Field(default_factory=list)
    summary: str = Field("", description="Résumé synthétique")
    metadata: Dict[str, str] = Field(default_factory=dict)
    raw_model_output: Optional[dict] = Field(
        None, description="Sortie brute du modèle (debug)"
    )







