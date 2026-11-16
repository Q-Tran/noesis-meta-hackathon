"""Value types shared across bounded knowledge bases."""
from enum import Enum


class DescriptionSource(str, Enum):
    llm = "llm"
    human = "human"
    imported = "imported"


class SourceType(str, Enum):
    internal_sql = "internal_sql"
    external = "external"
