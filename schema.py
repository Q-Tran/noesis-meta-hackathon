"""Compatibility facade for the bounded Noesis knowledge-base models.

New code should import from ``knowledge_bases.<domain>.models``. Existing flat
imports remain supported while callers migrate.
"""

from knowledge_bases.business_dictionary.models import (
    AlternateBusinessTerm,
    BusinessTerm,
)
from knowledge_bases.common import DescriptionSource, SourceType
from knowledge_bases.strategy.models import (
    Alternative,
    Decision,
    DecisionStatus,
    EdgeStatus,
    Goal,
    Kind,
    Stance,
)
from knowledge_bases.project_management.models import (
    Epic,
    Person,
    Project,
    ProjectStatus,
    Task,
    TaskStatus,
    UserStory,
)
from knowledge_bases.epistemology.models import FeedbackAnalysis, FeedbackClaim
from knowledge_bases.provenance.models import SourceNode, SourceReliability
from knowledge_bases.semantic_layer.models import ColumnNode, Metric, TableNode, ValueNode

__all__ = [
    "DescriptionSource", "SourceType", "BusinessTerm", "AlternateBusinessTerm",
    "TableNode", "ColumnNode", "ValueNode", "Metric", "SourceReliability",
    "SourceNode", "Stance", "Kind", "EdgeStatus",
    "DecisionStatus", "Goal", "Alternative", "Decision", "Person",
    "ProjectStatus", "Project", "Epic", "UserStory", "TaskStatus", "Task",
]
