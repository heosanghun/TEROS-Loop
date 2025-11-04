"""
TEROS-Loop 통합 시스템

3단계 자기 진화 메커니즘 통합
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .discrepancy_detector import DiscrepancyDetector, DiscrepancyCase
from .causal_analyzer import CausalAnalyzer, KnowledgeExtractor
from .validation_mechanism import ValidationMechanism, ProvisionalKnowledgeRule
from .ontology import EducationalKnowledgeOntology


class TEROSLoopStatus(BaseModel):
    """TEROS-Loop 상태"""
    loop_id: str
    case_id: str
    stage: str  # 'detection', 'analysis', 'validation', 'integration', 'completed'
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TEROSLoop:
    """TEROS-Loop 통합 시스템"""
    
    def __init__(
        self,
        mongodb_url: str,
        openai_api_key: Optional[str] = None,
        llama_model_path: Optional[str] = None,
        ontology_path: str = "./ontology/teros_ontology.owl"
    ):
        """
        Args:
            mongodb_url: MongoDB 연결 URL
            openai_api_key: OpenAI API 키
            llama_model_path: Llama 모델 경로
            ontology_path: 온톨로지 파일 경로
        """
        self.detector = DiscrepancyDetector(threshold=0.5)
        self.causal_analyzer = CausalAnalyzer(
            openai_api_key=openai_api_key,
            llama_model_path=llama_model_path
        )
        self.knowledge_extractor = KnowledgeExtractor(self.causal_analyzer)
        self.validation_mechanism = ValidationMechanism(mongodb_url)
        self.ontology = EducationalKnowledgeOntology(ontology_path)
        
        self.active_loops: Dict[str, TEROSLoopStatus] = {}
    
    async def process_discrepancy(
        self,
        student_id: str,
        prediction: Dict[str, Any],
        actual_result: Dict[str, Any],
        teacher_annotation: Optional[Dict[str, Any]] = None
    ) -> Optional[TEROSLoopStatus]:
        """
        불일치 사례 처리 (TEROS-Loop 전체 프로세스)
        
        Args:
            student_id: 학생 ID
            prediction: AI 예측 결과
            actual_result: 실제 결과
            teacher_annotation: 교사 어노테이션 (선택사항)
            
        Returns:
            TEROS-Loop 상태
        """
        # 1단계: 불일치 포착
        discrepancy_case = self.detector.detect_discrepancy(
            student_id=student_id,
            prediction=prediction,
            actual_result=actual_result,
            teacher_annotation=teacher_annotation
        )
        
        if not discrepancy_case:
            return None  # 불일치 없음
        
        # TEROS-Loop 상태 생성
        loop_id = f"loop_{datetime.now().strftime('%Y%m%d%H%M%S')}_{discrepancy_case.case_id}"
        loop_status = TEROSLoopStatus(
            loop_id=loop_id,
            case_id=discrepancy_case.case_id,
            stage="detection",
            status="processing"
        )
        self.active_loops[loop_id] = loop_status
        
        # 2단계: 원인 분석 및 지식 추출
        try:
            loop_status.stage = "analysis"
            loop_status.updated_at = datetime.now()
            
            discrepancy_dict = discrepancy_case.dict()
            knowledge_rule = await self.knowledge_extractor.extract_knowledge(discrepancy_dict)
            
            if not knowledge_rule:
                loop_status.status = "failed"
                loop_status.metadata["error"] = "Knowledge extraction failed"
                return loop_status
            
            # 3단계: 검증 메커니즘
            loop_status.stage = "validation"
            loop_status.updated_at = datetime.now()
            
            # 잠정적 지식 규칙 생성
            provisional_rule = ProvisionalKnowledgeRule(
                rule_id=knowledge_rule.rule_id,
                condition=knowledge_rule.condition,
                result=knowledge_rule.result,
                confidence_score=knowledge_rule.confidence,
                source_case_id=discrepancy_case.case_id,
                status="pending"
            )
            
            # 개념적 버퍼에 저장
            self.validation_mechanism.store_provisional_knowledge(provisional_rule)
            
            # 불일치 사례 상태 업데이트
            self.detector.update_case_status(discrepancy_case.case_id, "analyzing")
            
            loop_status.status = "completed"
            loop_status.stage = "validation"
            loop_status.metadata = {
                "rule_id": knowledge_rule.rule_id,
                "confidence": knowledge_rule.confidence
            }
            
            return loop_status
            
        except Exception as e:
            loop_status.status = "failed"
            loop_status.metadata["error"] = str(e)
            return loop_status
    
    async def update_rule_confidence(
        self,
        rule_id: str,
        observed_result: bool
    ) -> ProvisionalKnowledgeRule:
        """
        규칙 신뢰도 업데이트
        
        Args:
            rule_id: 규칙 ID
            observed_result: THEN 결과 관찰 여부
            
        Returns:
            업데이트된 규칙
        """
        return self.validation_mechanism.update_rule_confidence(
            rule_id,
            observed_result
        )
    
    def get_pending_deliberation_items(self) -> List[Dict[str, Any]]:
        """대기 중인 공동 심의 안건 조회"""
        items = self.validation_mechanism.get_pending_deliberation_items()
        return [item.dict() for item in items]
    
    def submit_teacher_response(
        self,
        item_id: str,
        response: str,  # 'approve', 'reject', 'modify'
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        교사 응답 제출 및 온톨로지 통합
        
        Args:
            item_id: 안건 ID
            response: 교사 응답
            feedback: 교사 피드백
            
        Returns:
            처리 결과
        """
        # 교사 응답 제출
        item = self.validation_mechanism.submit_teacher_response(
            item_id,
            response,
            feedback
        )
        
        # 승인된 경우 온톨로지 통합
        if response == "approve":
            rule = item.rule
            rule_dict = {
                "condition": rule.condition,
                "result": rule.result
            }
            
            # 온톨로지에 통합
            ontology_rule_id = self.ontology.integrate_knowledge_rule(rule_dict)
            
            return {
                "success": True,
                "item": item.dict(),
                "ontology_rule_id": ontology_rule_id,
                "message": "지식이 온톨로지에 통합되었습니다."
            }
        else:
            return {
                "success": True,
                "item": item.dict(),
                "message": f"교사 응답이 처리되었습니다: {response}"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """TEROS-Loop 통계"""
        all_rules = self.validation_mechanism.buffer.get_all_rules()
        
        stats = {
            "total_cases": len(self.detector.detected_cases),
            "pending_cases": len([c for c in self.detector.detected_cases if c.status == "pending"]),
            "analyzing_cases": len([c for c in self.detector.detected_cases if c.status == "analyzing"]),
            "resolved_cases": len([c for c in self.detector.detected_cases if c.status == "resolved"]),
            "total_rules": len(all_rules),
            "pending_rules": len([r for r in all_rules if r.status == "pending"]),
            "approved_rules": len([r for r in all_rules if r.status == "approved"]),
            "rejected_rules": len([r for r in all_rules if r.status == "rejected"]),
            "pending_deliberation": len(self.validation_mechanism.get_pending_deliberation_items()),
            "ontology_rules": len(self.ontology.get_knowledge_rules())
        }
        
        return stats

