"""
분석 API 라우트
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Any, Optional
import numpy as np

from app.analytics.fairness import FairnessEnhancementModule
from app.analytics.explainability import ExplainabilityModule
from app.analytics.context import ContextAwarenessModule, TeacherAnnotation, StudentMetacognitionReport
from app.analytics.analyzer import TalentDiagnosisAnalyzer

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

# 전역 인스턴스
fairness_module = FairnessEnhancementModule(input_dim=128, talent_categories=10, sensitive_categories=2)
explainability_module = ExplainabilityModule()
context_module = ContextAwarenessModule()
analyzer = TalentDiagnosisAnalyzer(
    fairness_module=fairness_module,
    explainability_module=explainability_module,
    context_module=context_module
)


@router.post("/talent-diagnosis")
async def diagnose_talent(
    student_id: str = Body(...),
    text_features: Optional[List[List[float]]] = Body(None),
    image_features: Optional[List[List[float]]] = Body(None),
    audio_features: Optional[List[List[float]]] = Body(None),
    video_features: Optional[List[List[float]]] = Body(None),
    base_features: Optional[List[float]] = Body(None)
):
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
    try:
        # NumPy 배열로 변환
        text_np = None if text_features is None else np.array(text_features)
        image_np = None if image_features is None else np.array(image_features)
        audio_np = None if audio_features is None else np.array(audio_features)
        video_np = None if video_features is None else np.array(video_features)
        base_np = None if base_features is None else np.array(base_features)
        
        # 재능 진단
        profile = analyzer.analyze(
            student_id=student_id,
            text_features=text_np,
            image_features=image_np,
            audio_features=audio_np,
            video_features=video_np,
            base_features=base_np
        )
        
        return {
            "success": True,
            "profile": profile.dict(),
            "message": "재능 진단이 완료되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def generate_explanation(
    text_data: Optional[Dict[str, Any]] = Body(None),
    image_data: Optional[Dict[str, Any]] = Body(None),
    audio_data: Optional[Dict[str, Any]] = Body(None),
    target: int = Body(0)
):
    """
    설명 생성
    
    Args:
        text_data: 텍스트 데이터
        image_data: 이미지 데이터
        audio_data: 오디오 데이터
        target: 타겟 클래스
        
    Returns:
        설명 결과
    """
    try:
        explanation = explainability_module.generate_explanation(
            text_data=text_data,
            image_data=image_data,
            audio_data=audio_data,
            target=target
        )
        
        return {
            "success": True,
            "explanation": explanation,
            "message": "설명이 생성되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual")
async def generate_counterfactual(
    original_input: List[float] = Body(...),
    target_class: int = Body(...),
    feature_names: List[str] = Body(...)
):
    """
    반사실적 설명 생성
    
    Args:
        original_input: 원본 입력
        target_class: 목표 클래스
        feature_names: 특징 이름 리스트
        
    Returns:
        반사실적 설명
    """
    try:
        original_np = np.array(original_input)
        explanation = explainability_module.generate_counterfactual_explanation(
            original_input=original_np,
            target_class=target_class,
            feature_names=feature_names
        )
        
        return {
            "success": True,
            "counterfactual": explanation,
            "message": "반사실적 설명이 생성되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/annotation")
async def add_teacher_annotation(
    annotation: TeacherAnnotation
):
    """
    교사 어노테이션 추가
    
    Args:
        annotation: 교사 어노테이션 객체
        
    Returns:
        추가된 어노테이션
    """
    try:
        added_annotation = context_module.add_teacher_annotation(annotation)
        
        return {
            "success": True,
            "annotation": added_annotation.dict(),
            "message": "교사 어노테이션이 추가되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/metacognition")
async def add_metacognition_report(
    report: StudentMetacognitionReport
):
    """
    학생 메타인지 리포트 추가
    
    Args:
        report: 메타인지 리포트 객체
        
    Returns:
        추가된 리포트
    """
    try:
        added_report = context_module.add_metacognition_report(report)
        
        return {
            "success": True,
            "report": added_report.dict(),
            "message": "메타인지 리포트가 추가되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{student_id}")
async def get_context(
    student_id: str,
    document_id: Optional[str] = None
):
    """
    학생의 맥락 정보 조회
    
    Args:
        student_id: 학생 ID
        document_id: 문서 ID (선택사항)
        
    Returns:
        맥락 정보
    """
    try:
        context = context_module.get_context_for_student(student_id, document_id)
        
        return {
            "success": True,
            "context": context,
            "message": "맥락 정보를 조회했습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/tags")
async def get_tags(
    category: Optional[str] = None
):
    """
    태그 목록 조회
    
    Args:
        category: 태그 카테고리 필터링
        
    Returns:
        태그 리스트
    """
    try:
        tags = context_module.get_tags(category)
        
        return {
            "success": True,
            "tags": [tag.dict() for tag in tags],
            "message": "태그 목록을 조회했습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fairness/evaluate")
async def evaluate_fairness(
    predictions: List[List[float]] = Body(...),
    sensitive_attributes: List[int] = Body(...),
    actual_labels: List[int] = Body(...)
):
    """
    공정성 평가
    
    Args:
        predictions: 예측 결과
        sensitive_attributes: 민감 정보
        actual_labels: 실제 레이블
        
    Returns:
        공정성 평가 결과
    """
    try:
        predictions_np = np.array(predictions)
        sensitive_np = np.array(sensitive_attributes)
        labels_np = np.array(actual_labels)
        
        fairness_result = fairness_module.evaluate_fairness(
            predictions_np,
            sensitive_np,
            labels_np
        )
        
        return {
            "success": True,
            "fairness": fairness_result,
            "message": "공정성 평가가 완료되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

