"""
TEROS-Loop 3단계: 신뢰도 강화 검증 메커니즘

개념적 버퍼, 베이즈 신뢰도 업데이트, 교사-AI 공동 심의
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.collection import Collection
import uuid


class ProvisionalKnowledgeRule(BaseModel):
    """잠정적 지식 규칙 (개념적 버퍼용)"""
    rule_id: str
    condition: Dict[str, Any]
    result: Dict[str, Any]
    confidence_score: float = 0.5  # 초기 신뢰도
    prior_probability: float = 0.5  # 사전 확률
    posterior_probability: float = 0.5  # 사후 확률
    validation_count: int = 0
    positive_count: int = 0  # IF 조건이 맞고 THEN 결과가 관찰된 횟수
    negative_count: int = 0  # IF 조건이 맞지만 THEN 결과가 관찰되지 않은 횟수
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # 'pending', 'validating', 'ready_for_review', 'approved', 'rejected', 'modified'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeliberationItem(BaseModel):
    """공동 심의 안건"""
    item_id: str
    rule_id: str
    rule: ProvisionalKnowledgeRule
    accumulated_evidence: List[Dict[str, Any]]  # 누적 사례 데이터
    confidence_score: float
    teacher_response: Optional[str] = None  # 'approve', 'reject', 'modify'
    teacher_feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConceptualBuffer:
    """개념적 버퍼 (MongoDB 기반)"""
    
    def __init__(self, mongodb_url: str, database_name: str = "teros_buffer"):
        """
        Args:
            mongodb_url: MongoDB 연결 URL
            database_name: 데이터베이스 이름
        """
        self.client = MongoClient(mongodb_url)
        self.db = self.client[database_name]
        self.collection: Collection = self.db["provisional_knowledge"]
    
    def store_rule(self, rule: ProvisionalKnowledgeRule) -> str:
        """
        잠정적 지식 규칙 저장
        
        Args:
            rule: 잠정적 지식 규칙
            
        Returns:
            규칙 ID
        """
        rule_dict = rule.dict()
        rule_dict["_id"] = rule.rule_id
        self.collection.insert_one(rule_dict)
        return rule.rule_id
    
    def get_rule(self, rule_id: str) -> Optional[ProvisionalKnowledgeRule]:
        """규칙 조회"""
        rule_dict = self.collection.find_one({"_id": rule_id})
        if rule_dict:
            rule_dict.pop("_id")
            return ProvisionalKnowledgeRule(**rule_dict)
        return None
    
    def update_rule(self, rule: ProvisionalKnowledgeRule):
        """규칙 업데이트"""
        rule_dict = rule.dict()
        rule_dict["updated_at"] = datetime.now()
        self.collection.update_one(
            {"_id": rule.rule_id},
            {"$set": rule_dict}
        )
    
    def get_rules_ready_for_review(self, threshold: float = 0.8) -> List[ProvisionalKnowledgeRule]:
        """검증 요청 준비된 규칙 조회"""
        rules_dict = self.collection.find({
            "status": "ready_for_review",
            "confidence_score": {"$gte": threshold}
        })
        return [ProvisionalKnowledgeRule(**r) for r in rules_dict if "_id" in r]
    
    def get_all_rules(self) -> List[ProvisionalKnowledgeRule]:
        """모든 규칙 조회"""
        rules_dict = self.collection.find({})
        rules = []
        for r in rules_dict:
            if "_id" in r:
                r.pop("_id")
                rules.append(ProvisionalKnowledgeRule(**r))
        return rules


class BayesianConfidenceUpdater:
    """베이즈 신뢰도 업데이트 시스템"""
    
    def update_confidence(
        self,
        rule: ProvisionalKnowledgeRule,
        observed_result: bool
    ) -> ProvisionalKnowledgeRule:
        """
        베이즈 정리를 통한 신뢰도 업데이트
        
        Args:
            rule: 잠정적 지식 규칙
            observed_result: THEN 결과가 관찰되었는지 여부
            
        Returns:
            업데이트된 규칙
        """
        # 검증 횟수 증가
        rule.validation_count += 1
        
        if observed_result:
            rule.positive_count += 1
        else:
            rule.negative_count += 1
        
        # 베이즈 정리 적용
        # P(H|E) = P(E|H) * P(H) / P(E)
        # P(H): 사전 확률 (prior)
        # P(E|H): 관찰된 결과가 주어진 경우 가설의 확률
        # P(H|E): 사후 확률 (posterior)
        
        # 간단한 베이즈 업데이트
        total_observations = rule.positive_count + rule.negative_count
        if total_observations > 0:
            # 관찰된 결과의 빈도
            likelihood = rule.positive_count / total_observations
            
            # 베이즈 업데이트 (단순화)
            prior = rule.prior_probability
            posterior = (likelihood * prior) / (
                likelihood * prior + (1 - likelihood) * (1 - prior)
            )
            
            rule.posterior_probability = posterior
            rule.confidence_score = posterior
        
        # 신뢰도가 임계치를 넘으면 검증 요청 준비
        if rule.confidence_score >= 0.8 and rule.status == "pending":
            rule.status = "ready_for_review"
        
        rule.updated_at = datetime.now()
        
        return rule


class ValidationMechanism:
    """검증 메커니즘"""
    
    def __init__(
        self,
        mongodb_url: str,
        confidence_threshold: float = 0.8
    ):
        """
        Args:
            mongodb_url: MongoDB 연결 URL
            confidence_threshold: 신뢰도 임계치
        """
        self.buffer = ConceptualBuffer(mongodb_url)
        self.confidence_updater = BayesianConfidenceUpdater()
        self.confidence_threshold = confidence_threshold
        self.deliberation_items: Dict[str, DeliberationItem] = {}
    
    def store_provisional_knowledge(
        self,
        rule: ProvisionalKnowledgeRule
    ) -> str:
        """
        잠정적 지식 규칙 저장 (개념적 버퍼에)
        
        Args:
            rule: 잠정적 지식 규칙
            
        Returns:
            규칙 ID
        """
        return self.buffer.store_rule(rule)
    
    def update_rule_confidence(
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
        rule = self.buffer.get_rule(rule_id)
        if not rule:
            raise ValueError(f"Rule not found: {rule_id}")
        
        updated_rule = self.confidence_updater.update_confidence(rule, observed_result)
        self.buffer.update_rule(updated_rule)
        
        # 검증 요청 준비된 경우 안건 생성
        if updated_rule.status == "ready_for_review":
            self._create_deliberation_item(updated_rule)
        
        return updated_rule
    
    def _create_deliberation_item(self, rule: ProvisionalKnowledgeRule):
        """공동 심의 안건 생성"""
        item_id = f"deliberation_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 누적 사례 데이터 수집
        accumulated_evidence = self._collect_accumulated_evidence(rule)
        
        item = DeliberationItem(
            item_id=item_id,
            rule_id=rule.rule_id,
            rule=rule,
            accumulated_evidence=accumulated_evidence,
            confidence_score=rule.confidence_score
        )
        
        self.deliberation_items[item_id] = item
    
    def _collect_accumulated_evidence(
        self,
        rule: ProvisionalKnowledgeRule
    ) -> List[Dict[str, Any]]:
        """누적 사례 데이터 수집"""
        # TODO: 실제 사례 데이터 수집 로직
        # 현재는 더미 데이터
        evidence = [
            {
                "case_id": f"case_{i}",
                "observed_result": True,
                "timestamp": datetime.now().isoformat()
            }
            for i in range(rule.validation_count)
        ]
        return evidence
    
    def get_pending_deliberation_items(self) -> List[DeliberationItem]:
        """대기 중인 공동 심의 안건 조회"""
        return [
            item for item in self.deliberation_items.values()
            if item.teacher_response is None
        ]
    
    def submit_teacher_response(
        self,
        item_id: str,
        response: str,  # 'approve', 'reject', 'modify'
        feedback: Optional[str] = None
    ) -> DeliberationItem:
        """
        교사 응답 제출
        
        Args:
            item_id: 안건 ID
            response: 교사 응답 ('approve', 'reject', 'modify')
            feedback: 교사 피드백 (선택사항)
            
        Returns:
            업데이트된 안건
        """
        if item_id not in self.deliberation_items:
            raise ValueError(f"Deliberation item not found: {item_id}")
        
        item = self.deliberation_items[item_id]
        item.teacher_response = response
        item.teacher_feedback = feedback
        item.reviewed_at = datetime.now()
        
        # 규칙 상태 업데이트
        rule = item.rule
        if response == "approve":
            rule.status = "approved"
        elif response == "reject":
            rule.status = "rejected"
        elif response == "modify":
            rule.status = "modified"
        
        self.buffer.update_rule(rule)
        
        return item
    
    def get_approved_rules(self) -> List[ProvisionalKnowledgeRule]:
        """승인된 규칙 조회"""
        all_rules = self.buffer.get_all_rules()
        return [rule for rule in all_rules if rule.status == "approved"]

