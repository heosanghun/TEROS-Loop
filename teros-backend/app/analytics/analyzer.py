"""
재능 진단 분석기 (Talent Diagnosis Analyzer)

멀티모달 데이터를 통합하여 재능 프로파일 생성 및 진로 추천
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .fairness import FairnessEnhancementModule
from .explainability import ExplainabilityModule
from .context import ContextAwarenessModule


class TalentCategory(BaseModel):
    """재능 카테고리"""
    category_id: str
    category_name: str
    score: float
    confidence: float
    evidence: List[str] = Field(default_factory=list)


class TalentProfile(BaseModel):
    """재능 프로파일"""
    student_id: str
    talents: List[TalentCategory] = Field(default_factory=list)
    overall_score: float
    top_talents: List[str] = Field(default_factory=list)
    career_recommendations: List[str] = Field(default_factory=list)
    learning_path: List[str] = Field(default_factory=list)
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TalentDiagnosisAnalyzer:
    """재능 진단 분석기"""
    
    # 재능 카테고리 정의
    TALENT_CATEGORIES = [
        {"id": "linguistic", "name": "언어·문학", "description": "언어 능력, 문학적 감수성"},
        {"id": "logical", "name": "논리·수학", "description": "논리적 사고, 수학적 능력"},
        {"id": "spatial", "name": "공간·시각", "description": "공간 인지, 시각적 표현"},
        {"id": "musical", "name": "음악·리듬", "description": "음악적 감각, 리듬감"},
        {"id": "bodily", "name": "신체·운동", "description": "신체 조절, 운동 능력"},
        {"id": "interpersonal", "name": "대인관계", "description": "협업, 소통 능력"},
        {"id": "intrapersonal", "name": "자기이해", "description": "성찰, 자기 인식"},
        {"id": "naturalistic", "name": "자연탐구", "description": "자연 관찰, 과학적 탐구"},
        {"id": "creative", "name": "창의성", "description": "창의적 사고, 혁신"},
        {"id": "technical", "name": "기술·공학", "description": "기술 이해, 공학적 능력"},
    ]
    
    # 진로 추천 매핑
    CAREER_MAPPING = {
        "linguistic": ["언어치료사", "번역가", "작가", "언론인"],
        "logical": ["수학자", "프로그래머", "엔지니어", "연구원"],
        "spatial": ["건축가", "디자이너", "예술가", "애니메이션 제작자"],
        "musical": ["음악가", "작곡가", "음향 엔지니어", "연출가"],
        "bodily": ["운동선수", "무용가", "물리치료사", "체육교사"],
        "interpersonal": ["상담사", "교사", "경영자", "인사담당자"],
        "intrapersonal": ["심리학자", "철학자", "작가", "상담사"],
        "naturalistic": ["생물학자", "환경과학자", "수의사", "농학자"],
        "creative": ["예술가", "디자이너", "창업가", "기획자"],
        "technical": ["엔지니어", "프로그래머", "과학자", "기술자"],
    }
    
    def __init__(
        self,
        fairness_module: Optional[FairnessEnhancementModule] = None,
        explainability_module: Optional[ExplainabilityModule] = None,
        context_module: Optional[ContextAwarenessModule] = None
    ):
        """
        Args:
            fairness_module: 공정성 강화 모듈
            explainability_module: 설명가능성 모듈
            context_module: 맥락 이해 모듈
        """
        self.fairness_module = fairness_module
        self.explainability_module = explainability_module
        self.context_module = context_module
    
    def analyze(
        self,
        student_id: str,
        text_features: Optional[np.ndarray] = None,
        image_features: Optional[np.ndarray] = None,
        audio_features: Optional[np.ndarray] = None,
        video_features: Optional[np.ndarray] = None,
        base_features: Optional[np.ndarray] = None
    ) -> TalentProfile:
        """
        재능 진단 분석
        
        Args:
            student_id: 학생 ID
            text_features: 텍스트 특징 벡터
            image_features: 이미지 특징 벡터
            audio_features: 오디오 특징 벡터
            video_features: 비디오 특징 벡터
            base_features: 기본 특징 벡터
            
        Returns:
            재능 프로파일
        """
        # 멀티모달 특징 통합
        integrated_features = self._integrate_multimodal_features(
            text_features,
            image_features,
            audio_features,
            video_features,
            base_features
        )
        
        # 맥락 정보 통합
        if self.context_module:
            integrated_features = self.context_module.integrate_context_to_features(
                integrated_features,
                student_id
            )
        
        # 재능 점수 계산
        talent_scores = self._calculate_talent_scores(integrated_features)
        
        # 재능 카테고리 생성
        talents = self._create_talent_categories(talent_scores)
        
        # 전체 점수 계산
        overall_score = np.mean([t.score for t in talents])
        
        # 상위 재능 추출
        top_talents = sorted(talents, key=lambda x: x.score, reverse=True)[:3]
        top_talent_ids = [t.category_id for t in top_talents]
        
        # 진로 추천 생성
        career_recommendations = self._generate_career_recommendations(top_talent_ids)
        
        # 학습 경로 생성
        learning_path = self._generate_learning_path(top_talent_ids)
        
        return TalentProfile(
            student_id=student_id,
            talents=talents,
            overall_score=float(overall_score),
            top_talents=top_talent_ids,
            career_recommendations=career_recommendations,
            learning_path=learning_path,
            created_at=str(datetime.now()),
            metadata={}
        )
    
    def _integrate_multimodal_features(
        self,
        text_features: Optional[np.ndarray],
        image_features: Optional[np.ndarray],
        audio_features: Optional[np.ndarray],
        video_features: Optional[np.ndarray],
        base_features: Optional[np.ndarray]
    ) -> np.ndarray:
        """멀티모달 특징 통합"""
        features_list = []
        
        if base_features is not None:
            features_list.append(base_features)
        
        if text_features is not None:
            if isinstance(text_features, list):
                text_avg = np.mean(text_features, axis=0)
            else:
                text_avg = text_features
            features_list.append(text_avg)
        
        if image_features is not None:
            if isinstance(image_features, list):
                image_avg = np.mean(image_features, axis=0)
            else:
                image_avg = image_features
            features_list.append(image_avg)
        
        if audio_features is not None:
            if isinstance(audio_features, list):
                audio_avg = np.mean(audio_features, axis=0)
            else:
                audio_avg = audio_features
            features_list.append(audio_avg)
        
        if video_features is not None:
            if isinstance(video_features, list):
                video_avg = np.mean(video_features, axis=0)
            else:
                video_avg = video_features
            features_list.append(video_avg)
        
        if not features_list:
            # 기본 특징이 없으면 더미 특징 생성
            return np.random.rand(128)
        
        # 특징 통합 (평균 또는 연결)
        integrated = np.concatenate(features_list)
        
        # 정규화
        if integrated.max() > 1.0 or integrated.min() < 0.0:
            integrated = (integrated - integrated.min()) / (integrated.max() - integrated.min() + 1e-8)
        
        return integrated
    
    def _calculate_talent_scores(self, features: np.ndarray) -> Dict[str, float]:
        """재능 점수 계산"""
        # TODO: 실제 ML 모델 사용
        # 현재는 특징 벡터를 기반으로 더미 점수 생성
        
        # 특징 벡터를 재능 카테고리 수로 매핑
        num_talents = len(self.TALENT_CATEGORIES)
        
        # 특징 벡터를 재능 점수로 변환 (간단한 선형 변환)
        if len(features) >= num_talents:
            # 특징을 재능 수만큼 샘플링
            sampled_features = features[:num_talents]
        else:
            # 특징이 부족하면 반복
            repeat_factor = (num_talents // len(features)) + 1
            sampled_features = np.tile(features, repeat_factor)[:num_talents]
        
        # 0-100 점수로 변환
        scores = (sampled_features * 100).clip(0, 100)
        
        # 재능 카테고리별 점수 매핑
        talent_scores = {}
        for i, category in enumerate(self.TALENT_CATEGORIES):
            talent_scores[category["id"]] = float(scores[i])
        
        return talent_scores
    
    def _create_talent_categories(
        self,
        talent_scores: Dict[str, float]
    ) -> List[TalentCategory]:
        """재능 카테고리 생성"""
        talents = []
        
        for category in self.TALENT_CATEGORIES:
            score = talent_scores.get(category["id"], 0.0)
            confidence = min(score / 100.0, 1.0)  # 신뢰도는 점수 기반
            
            # 근거 생성
            evidence = []
            if score > 70:
                evidence.append("높은 점수")
            if score > 50:
                evidence.append("중간 이상 점수")
            
            talents.append(TalentCategory(
                category_id=category["id"],
                category_name=category["name"],
                score=score,
                confidence=confidence,
                evidence=evidence
            ))
        
        return talents
    
    def _generate_career_recommendations(
        self,
        top_talent_ids: List[str]
    ) -> List[str]:
        """진로 추천 생성"""
        recommendations = []
        
        for talent_id in top_talent_ids:
            if talent_id in self.CAREER_MAPPING:
                recommendations.extend(self.CAREER_MAPPING[talent_id])
        
        # 중복 제거 및 상위 5개 선택
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]
    
    def _generate_learning_path(
        self,
        top_talent_ids: List[str]
    ) -> List[str]:
        """학습 경로 생성"""
        learning_paths = {
            "linguistic": ["독서 활동", "작문 연습", "토론 참여"],
            "logical": ["수학 문제 해결", "논리적 사고 훈련", "프로그래밍"],
            "spatial": ["그림 그리기", "3D 모델링", "디자인 프로젝트"],
            "musical": ["악기 연주", "작곡 연습", "음악 이론 학습"],
            "bodily": ["운동 활동", "무용 연습", "신체 표현"],
            "interpersonal": ["팀 프로젝트", "봉사 활동", "협업 연습"],
            "intrapersonal": ["일기 쓰기", "명상", "자기 성찰"],
            "naturalistic": ["자연 관찰", "실험 활동", "환경 탐구"],
            "creative": ["창의적 프로젝트", "아이디어 발상", "혁신적 사고"],
            "technical": ["기술 프로젝트", "엔지니어링", "실험 설계"],
        }
        
        path = []
        for talent_id in top_talent_ids:
            if talent_id in learning_paths:
                path.extend(learning_paths[talent_id])
        
        return list(dict.fromkeys(path))[:10]

