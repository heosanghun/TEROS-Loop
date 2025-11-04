"""
자가 발전 엔진 계층 (Self-Evolving Engine Layer)

TEROS-Loop를 통한 자기 진화 메커니즘
"""

from .teros_loop import TEROSLoop
from .discrepancy_detector import DiscrepancyDetector
from .causal_analyzer import CausalAnalyzer
from .knowledge_extractor import KnowledgeExtractor
from .validation_mechanism import ValidationMechanism
from .ontology import EducationalKnowledgeOntology

__all__ = [
    "TEROSLoop",
    "DiscrepancyDetector",
    "CausalAnalyzer",
    "KnowledgeExtractor",
    "ValidationMechanism",
    "EducationalKnowledgeOntology"
]

