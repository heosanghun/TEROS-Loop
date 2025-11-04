"""
분석 모듈 테스트
"""
import pytest
import numpy as np
from app.analytics.fairness import FairnessEnhancementModule
from app.analytics.explainability import ExplainabilityModule
from app.analytics.context import ContextAwarenessModule
from app.analytics.analyzer import TalentDiagnosisAnalyzer


@pytest.fixture
def fairness_module():
    """공정성 강화 모듈 픽스처"""
    return FairnessEnhancementModule(
        input_dim=128,
        talent_categories=10,
        sensitive_categories=2
    )


@pytest.fixture
def context_module():
    """맥락 이해 모듈 픽스처"""
    return ContextAwarenessModule()


def test_fairness_evaluation(fairness_module):
    """공정성 평가 테스트"""
    # 더미 예측 및 실제 레이블
    predictions = np.random.rand(100, 10)
    sensitive_attributes = np.random.randint(0, 2, 100)
    actual_labels = np.random.randint(0, 10, 100)
    
    result = fairness_module.evaluate_fairness(
        predictions,
        sensitive_attributes,
        actual_labels
    )
    
    assert "chi_squared" in result
    assert "p_value" in result
    assert "is_fair" in result
    assert isinstance(result["is_fair"], bool)


def test_context_tagging(context_module):
    """맥락 태깅 테스트"""
    tags = context_module.get_tags()
    assert len(tags) > 0
    
    # 카테고리별 필터링
    condition_tags = context_module.get_tags(category="condition")
    assert all(tag.tag_category == "condition" for tag in condition_tags)


def test_analyzer_integration():
    """재능 진단 분석기 통합 테스트"""
    analyzer = TalentDiagnosisAnalyzer()
    
    # 더미 특징 벡터
    text_features = np.random.rand(5, 768)
    image_features = np.random.rand(3, 768)
    
    profile = analyzer.analyze(
        student_id="test_student_001",
        text_features=text_features,
        image_features=image_features
    )
    
    assert profile.student_id == "test_student_001"
    assert len(profile.talents) == 10
    assert profile.overall_score >= 0
    assert len(profile.top_talents) <= 3
    assert len(profile.career_recommendations) > 0

