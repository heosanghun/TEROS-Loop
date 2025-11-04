"""
데이터 통합 계층 (Data Integration Layer)

멀티모달 데이터 수집 및 전처리 파이프라인
"""

from .collector import MultimodalDataCollector
from .preprocessor import DataPreprocessor

__all__ = ["MultimodalDataCollector", "DataPreprocessor"]

