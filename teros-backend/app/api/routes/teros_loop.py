"""
TEROS-Loop API 라우트
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Any, Optional
import os

from app.engine.teros_loop import TEROSLoop
from app.engine.validation_mechanism import ProvisionalKnowledgeRule

router = APIRouter(prefix="/api/v1/teros-loop", tags=["teros-loop"])

# 환경 변수에서 설정 가져오기
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://teros:teros_password@localhost:27017/teros_buffer?authSource=admin")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")

# 전역 인스턴스
teros_loop = TEROSLoop(
    mongodb_url=MONGODB_URL,
    openai_api_key=OPENAI_API_KEY,
    llama_model_path=LLAMA_MODEL_PATH
)


@router.post("/process-discrepancy")
async def process_discrepancy(
    student_id: str = Body(...),
    prediction: Dict[str, Any] = Body(...),
    actual_result: Dict[str, Any] = Body(...),
    teacher_annotation: Optional[Dict[str, Any]] = Body(None)
):
    """
    불일치 사례 처리 (TEROS-Loop 전체 프로세스)
    
    Args:
        student_id: 학생 ID
        prediction: AI 예측 결과
        actual_result: 실제 결과
        teacher_annotation: 교사 어노테이션 (선택사항)
        
    Returns:
        TEROS-Loop 처리 결과
    """
    try:
        loop_status = await teros_loop.process_discrepancy(
            student_id=student_id,
            prediction=prediction,
            actual_result=actual_result,
            teacher_annotation=teacher_annotation
        )
        
        if not loop_status:
            return {
                "success": False,
                "message": "불일치가 감지되지 않았습니다."
            }
        
        return {
            "success": True,
            "loop_status": loop_status.dict(),
            "message": "TEROS-Loop가 처리되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-confidence")
async def update_confidence(
    rule_id: str = Body(...),
    observed_result: bool = Body(...)
):
    """
    규칙 신뢰도 업데이트
    
    Args:
        rule_id: 규칙 ID
        observed_result: THEN 결과 관찰 여부
        
    Returns:
        업데이트된 규칙
    """
    try:
        updated_rule = await teros_loop.update_rule_confidence(
            rule_id,
            observed_result
        )
        
        return {
            "success": True,
            "rule": updated_rule.dict(),
            "message": "신뢰도가 업데이트되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deliberation/pending")
async def get_pending_deliberation():
    """대기 중인 공동 심의 안건 조회"""
    try:
        items = teros_loop.get_pending_deliberation_items()
        
        return {
            "success": True,
            "items": items,
            "count": len(items),
            "message": "대기 중인 안건을 조회했습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deliberation/respond")
async def submit_teacher_response(
    item_id: str = Body(...),
    response: str = Body(...),  # 'approve', 'reject', 'modify'
    feedback: Optional[str] = Body(None)
):
    """
    교사 응답 제출
    
    Args:
        item_id: 안건 ID
        response: 교사 응답
        feedback: 교사 피드백
        
    Returns:
        처리 결과
    """
    try:
        result = teros_loop.submit_teacher_response(
            item_id,
            response,
            feedback
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics():
    """TEROS-Loop 통계"""
    try:
        stats = teros_loop.get_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "message": "통계를 조회했습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cases/pending")
async def get_pending_cases():
    """대기 중인 불일치 사례 조회"""
    try:
        cases = teros_loop.detector.get_pending_cases()
        
        return {
            "success": True,
            "cases": [case.dict() for case in cases],
            "count": len(cases),
            "message": "대기 중인 불일치 사례를 조회했습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

