"""
TEROS-Loop 1단계: 예측-결과 불일치 자동 포착 엔진

KL Divergence 기반 불일치 감지
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from scipy import stats
from scipy.special import kl_div


class DiscrepancyCase(BaseModel):
    """불일치 사례"""
    case_id: str
    student_id: str
    prediction: Dict[str, Any]  # AI 예측 결과
    actual_result: Dict[str, Any]  # 실제 결과
    discrepancy_type: str  # 'prediction_mismatch', 'teacher_annotation_conflict'
    kl_divergence: float
    threshold: float = 0.5
    detected_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # 'pending', 'analyzing', 'resolved'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DiscrepancyDetector:
    """불일치 포착 엔진"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: KL Divergence 임계값
        """
        self.threshold = threshold
        self.detected_cases: List[DiscrepancyCase] = []
    
    def detect_discrepancy(
        self,
        student_id: str,
        prediction: Dict[str, Any],
        actual_result: Dict[str, Any],
        teacher_annotation: Optional[Dict[str, Any]] = None
    ) -> Optional[DiscrepancyCase]:
        """
        불일치 사례 감지
        
        Args:
            student_id: 학생 ID
            prediction: AI 예측 결과
            actual_result: 실제 결과
            teacher_annotation: 교사 어노테이션 (선택사항)
            
        Returns:
            불일치 사례 (감지된 경우), None (감지되지 않은 경우)
        """
        # 예측 분포와 실제 결과 분포 계산
        pred_distribution = self._extract_distribution(prediction)
        actual_distribution = self._extract_distribution(actual_result)
        
        # KL Divergence 계산
        kl_divergence = self._calculate_kl_divergence(
            pred_distribution,
            actual_distribution
        )
        
        # 교사 어노테이션 충돌 확인
        teacher_conflict = False
        if teacher_annotation:
            teacher_conflict = self._check_teacher_conflict(
                prediction,
                teacher_annotation
            )
        
        # 불일치 감지 조건
        is_discrepancy = (
            kl_divergence > self.threshold or
            teacher_conflict
        )
        
        if is_discrepancy:
            discrepancy_type = (
                "teacher_annotation_conflict" if teacher_conflict
                else "prediction_mismatch"
            )
            
            case = DiscrepancyCase(
                case_id=f"case_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.detected_cases)}",
                student_id=student_id,
                prediction=prediction,
                actual_result=actual_result,
                discrepancy_type=discrepancy_type,
                kl_divergence=kl_divergence,
                threshold=self.threshold,
                status="pending",
                metadata={
                    "teacher_annotation": teacher_annotation,
                    "pred_distribution": pred_distribution.tolist(),
                    "actual_distribution": actual_distribution.tolist()
                }
            )
            
            self.detected_cases.append(case)
            return case
        
        return None
    
    def _extract_distribution(self, data: Dict[str, Any]) -> np.ndarray:
        """분포 추출"""
        # 예측 결과에서 확률 분포 추출
        if "probabilities" in data:
            return np.array(data["probabilities"])
        elif "scores" in data:
            scores = np.array(data["scores"])
            # 정규화하여 확률 분포로 변환
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum()
        elif "talent_categories" in data:
            # 재능 카테고리 점수에서 분포 생성
            talents = data["talent_categories"]
            scores = np.array([t.get("score", 0.0) for t in talents])
            scores = scores / scores.sum() if scores.sum() > 0 else scores
            return scores
        else:
            # 기본 분포 (균등 분포)
            return np.ones(10) / 10
    
    def _calculate_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        KL Divergence 계산
        
        KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
        """
        # 0으로 나누기 방지
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # 정규화
        p = p / p.sum()
        q = q / q.sum()
        
        # KL Divergence 계산
        kl = np.sum(p * np.log(p / q))
        
        return float(kl)
    
    def _check_teacher_conflict(
        self,
        prediction: Dict[str, Any],
        teacher_annotation: Dict[str, Any]
    ) -> bool:
        """교사 어노테이션 충돌 확인"""
        # AI 예측과 교사 어노테이션 비교
        # 예: AI가 '논리적 재능'으로 예측했지만 교사가 '창의적 재능' 태그를 달았을 때
        
        if "tags" in teacher_annotation:
            teacher_tags = teacher_annotation["tags"]
            
            # AI 예측의 상위 재능 추출
            if "talent_categories" in prediction:
                top_talents = sorted(
                    prediction["talent_categories"],
                    key=lambda x: x.get("score", 0.0),
                    reverse=True
                )[:3]
                predicted_talent_ids = [t.get("category_id", "") for t in top_talents]
                
                # 태그와 재능 카테고리 매핑 (간단한 예시)
                tag_talent_mapping = {
                    "acad_confusion": "logical",
                    "acad_high_interest": "creative",
                    "pers_stage_fright": "interpersonal",
                    # ... 더 많은 매핑 필요
                }
                
                teacher_talent_ids = [
                    tag_talent_mapping.get(tag, "") for tag in teacher_tags
                    if tag in tag_talent_mapping
                ]
                
                # 충돌 확인: 교사 태그와 AI 예측이 일치하지 않음
                if teacher_talent_ids and predicted_talent_ids:
                    if not any(tid in predicted_talent_ids for tid in teacher_talent_ids):
                        return True
        
        # 자유 텍스트 주석 확인
        if "free_text" in teacher_annotation:
            free_text = teacher_annotation["free_text"].lower()
            # 부정적 표현이 있으면 충돌 가능성
            negative_keywords = ["틀림", "아님", "다름", "아니", "오류"]
            if any(keyword in free_text for keyword in negative_keywords):
                return True
        
        return False
    
    def get_pending_cases(self) -> List[DiscrepancyCase]:
        """대기 중인 불일치 사례 조회"""
        return [case for case in self.detected_cases if case.status == "pending"]
    
    def update_case_status(self, case_id: str, status: str):
        """사례 상태 업데이트"""
        for case in self.detected_cases:
            if case.case_id == case_id:
                case.status = status
                break

