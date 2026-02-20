from typing import List
from pydantic import BaseModel, Field, field_validator


class UserStory(BaseModel):
    id: str
    title: str
    as_a: str
    i_want: str
    so_that: str
    acceptance_criteria: List[str] = Field(default_factory=list)
    story_points: int
    priority: str        # Must | Should | Could
    business_value: int  # 1 à 10
    risk: str            # low | medium | high
    dependencies: List[str] = Field(default_factory=list)

    @field_validator("story_points")
    @classmethod
    def validate_fibonacci(cls, v: int) -> int:
        allowed = {1, 2, 3, 5, 8, 13, 21}
        if v not in allowed:
            raise ValueError(
                f"story_points doit appartenir à la suite Fibonacci {allowed}, reçu : {v}"
            )
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"Must", "Should", "Could"}
        if v not in allowed:
            raise ValueError(
                f"priority doit être Must | Should | Could, reçu : {v}"
            )
        return v

    @field_validator("risk")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v.lower() not in allowed:
            raise ValueError(
                f"risk doit être low | medium | high, reçu : {v}"
            )
        return v.lower()

    @field_validator("business_value")
    @classmethod
    def validate_business_value(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError(
                f"business_value doit être entre 1 et 10, reçu : {v}"
            )
        return v


class Epic(BaseModel):
    id: str
    name: str
    description: str
    user_stories: List[str] = Field(default_factory=list)


class Sprint(BaseModel):
    id: str
    user_stories: List[str] = Field(default_factory=list)
    # capacity_points = somme réelle des story_points du sprint
    # Pas de plafond strict ici : la validation métier est faite dans agent.py
    capacity_points: int = Field(ge=0)


class RoadmapItem(BaseModel):
    quarter: str   # ex: Q1, Q2, Q3, Q4
    epics: List[str] = Field(default_factory=list)


class ProductOwnerOutput(BaseModel):
    vision: str
    epics: List[Epic] = Field(default_factory=list)
    user_stories: List[UserStory] = Field(default_factory=list)
    sprints: List[Sprint] = Field(default_factory=list)
    roadmap: List[RoadmapItem] = Field(default_factory=list)