"""
TEROS-Loop 2단계: LLM 앙상블 기반 원인 분석 및 잠정적 지식 추출

불일치 사례의 근본 원인 분석
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import openai
from openai import OpenAI


class Hypothesis(BaseModel):
    """가설"""
    hypothesis_id: str
    llm_name: str  # 'gpt-4', 'llama-3', etc.
    content: str
    confidence: float
    reasoning: str
    evidence: List[str] = Field(default_factory=list)


class KnowledgeRule(BaseModel):
    """잠정적 지식 규칙 (IF-THEN 형태)"""
    rule_id: str
    condition: Dict[str, Any]  # IF 조건
    result: Dict[str, Any]  # THEN 결과
    confidence: float = 0.5  # 초기 신뢰도
    source_case_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    validation_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CausalAnalyzer:
    """원인 분석 엔진"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        llama_model_path: Optional[str] = None
    ):
        """
        Args:
            openai_api_key: OpenAI API 키
            llama_model_path: Llama 모델 경로 (선택사항)
        """
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.llama_model_path = llama_model_path
        self.hypotheses: List[Hypothesis] = []
        self.knowledge_rules: List[KnowledgeRule] = []
    
    async def analyze_discrepancy(
        self,
        discrepancy_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        불일치 사례 원인 분석
        
        Args:
            discrepancy_case: 불일치 사례 데이터
            
        Returns:
            분석 결과 (가설 리스트, 잠정적 지식 규칙)
        """
        # 각 LLM에 독립적 가설 생성 요청
        hypotheses = []
        
        # GPT-4 가설 생성
        if self.openai_client:
            gpt_hypothesis = await self._generate_gpt_hypothesis(discrepancy_case)
            if gpt_hypothesis:
                hypotheses.append(gpt_hypothesis)
        
        # Llama 3 가설 생성 (로컬 모델)
        if self.llama_model_path:
            llama_hypothesis = await self._generate_llama_hypothesis(discrepancy_case)
            if llama_hypothesis:
                hypotheses.append(llama_hypothesis)
        
        # 가설 종합 및 분석
        synthesized_hypothesis = self._synthesize_hypotheses(hypotheses)
        
        # 잠정적 지식 규칙 생성
        knowledge_rule = self._extract_knowledge_rule(
            synthesized_hypothesis,
            discrepancy_case.get("case_id", "")
        )
        
        if knowledge_rule:
            self.knowledge_rules.append(knowledge_rule)
        
        return {
            "hypotheses": [h.dict() for h in hypotheses],
            "synthesized_hypothesis": synthesized_hypothesis.dict() if synthesized_hypothesis else None,
            "knowledge_rule": knowledge_rule.dict() if knowledge_rule else None
        }
    
    async def _generate_gpt_hypothesis(
        self,
        discrepancy_case: Dict[str, Any]
    ) -> Optional[Hypothesis]:
        """GPT-4 가설 생성"""
        if not self.openai_client:
            return None
        
        try:
            prompt = self._construct_analysis_prompt(discrepancy_case)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 교육 AI 시스템의 불일치 사례를 분석하는 전문가입니다. "
                     "불일치의 근본 원인을 분석하고 가설을 제시하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # 가설 파싱
            hypothesis = self._parse_hypothesis(content, "gpt-4")
            
            return hypothesis
        except Exception as e:
            print(f"GPT-4 가설 생성 실패: {e}")
            return None
    
    async def _generate_llama_hypothesis(
        self,
        discrepancy_case: Dict[str, Any]
    ) -> Optional[Hypothesis]:
        """Llama 3 가설 생성"""
        # TODO: 실제 Llama 3 모델 통합
        # 현재는 더미 가설 반환
        prompt = self._construct_analysis_prompt(discrepancy_case)
        
        # 더미 가설 생성
        hypothesis = Hypothesis(
            hypothesis_id=f"llama_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            llm_name="llama-3",
            content="더미 가설: 데이터 편향이나 맥락 정보 부족이 원인일 수 있습니다.",
            confidence=0.6,
            reasoning="Llama 모델 분석 결과",
            evidence=[]
        )
        
        return hypothesis
    
    def _construct_analysis_prompt(self, discrepancy_case: Dict[str, Any]) -> str:
        """원인 분석 프롬프트 구성"""
        prompt = f"""
불일치 사례 분석:

학생 ID: {discrepancy_case.get('student_id', 'N/A')}
불일치 타입: {discrepancy_case.get('discrepancy_type', 'N/A')}
KL Divergence: {discrepancy_case.get('kl_divergence', 'N/A')}

AI 예측:
{json.dumps(discrepancy_case.get('prediction', {}), indent=2, ensure_ascii=False)}

실제 결과:
{json.dumps(discrepancy_case.get('actual_result', {}), indent=2, ensure_ascii=False)}

교사 어노테이션:
{json.dumps(discrepancy_case.get('metadata', {}).get('teacher_annotation', {}), indent=2, ensure_ascii=False)}

위 불일치 사례의 근본 원인을 분석하고, "IF [조건] THEN [결과]" 형태의 지식 규칙을 제시하세요.
"""
        return prompt
    
    def _parse_hypothesis(self, content: str, llm_name: str) -> Hypothesis:
        """가설 파싱"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        hypothesis = Hypothesis(
            hypothesis_id=f"{llm_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            llm_name=llm_name,
            content=content[:500],  # 처음 500자만
            confidence=0.7,  # 기본 신뢰도
            reasoning=content,
            evidence=[]
        )
        
        return hypothesis
    
    def _synthesize_hypotheses(
        self,
        hypotheses: List[Hypothesis]
    ) -> Optional[Hypothesis]:
        """가설 종합"""
        if not hypotheses:
            return None
        
        if len(hypotheses) == 1:
            return hypotheses[0]
        
        # 가설들의 공통점과 차이점 분석
        common_themes = []
        all_content = " ".join([h.content for h in hypotheses])
        
        # 간단한 종합 (실제로는 더 정교한 분석 필요)
        synthesized_content = f"종합 분석: {all_content[:300]}"
        
        # 평균 신뢰도
        avg_confidence = sum(h.confidence for h in hypotheses) / len(hypotheses)
        
        synthesized = Hypothesis(
            hypothesis_id=f"synthesized_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            llm_name="ensemble",
            content=synthesized_content,
            confidence=avg_confidence,
            reasoning="앙상블 종합 결과",
            evidence=[h.hypothesis_id for h in hypotheses]
        )
        
        return synthesized
    
    def _extract_knowledge_rule(
        self,
        hypothesis: Hypothesis,
        case_id: str
    ) -> Optional[KnowledgeRule]:
        """잠정적 지식 규칙 추출"""
        if not hypothesis:
            return None
        
        # 가설 내용에서 IF-THEN 규칙 추출
        # 간단한 패턴 매칭 (실제로는 더 정교한 NLP 필요)
        content = hypothesis.content.lower()
        
        # IF-THEN 패턴 찾기
        if "if" in content and "then" in content:
            parts = content.split("then")
            if len(parts) >= 2:
                condition_text = parts[0].replace("if", "").strip()
                result_text = parts[1].strip()
                
                # 조건과 결과를 구조화
                condition = self._parse_condition(condition_text)
                result = self._parse_result(result_text)
                
                rule = KnowledgeRule(
                    rule_id=f"rule_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    condition=condition,
                    result=result,
                    confidence=hypothesis.confidence,
                    source_case_id=case_id,
                    metadata={
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "reasoning": hypothesis.reasoning
                    }
                )
                
                return rule
        
        # IF-THEN 패턴이 없으면 기본 규칙 생성
        condition = {
            "type": "general",
            "description": hypothesis.content[:200]
        }
        result = {
            "type": "prediction_adjustment",
            "description": "예측 결과 조정 필요"
        }
        
        rule = KnowledgeRule(
            rule_id=f"rule_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            condition=condition,
            result=result,
            confidence=hypothesis.confidence,
            source_case_id=case_id,
            metadata={
                "hypothesis_id": hypothesis.hypothesis_id,
                "reasoning": hypothesis.reasoning
            }
        )
        
        return rule
    
    def _parse_condition(self, condition_text: str) -> Dict[str, Any]:
        """조건 파싱"""
        # 간단한 파싱 (실제로는 더 정교한 NLP 필요)
        return {
            "type": "text_condition",
            "description": condition_text[:200],
            "keywords": condition_text.split()[:10]
        }
    
    def _parse_result(self, result_text: str) -> Dict[str, Any]:
        """결과 파싱"""
        # 간단한 파싱
        return {
            "type": "text_result",
            "description": result_text[:200],
            "keywords": result_text.split()[:10]
        }


class KnowledgeExtractor:
    """지식 추출기"""
    
    def __init__(self, causal_analyzer: CausalAnalyzer):
        """
        Args:
            causal_analyzer: 원인 분석 엔진
        """
        self.causal_analyzer = causal_analyzer
    
    async def extract_knowledge(
        self,
        discrepancy_case: Dict[str, Any]
    ) -> Optional[KnowledgeRule]:
        """
        불일치 사례에서 지식 추출
        
        Args:
            discrepancy_case: 불일치 사례 데이터
            
        Returns:
            잠정적 지식 규칙
        """
        # 원인 분석
        analysis_result = await self.causal_analyzer.analyze_discrepancy(discrepancy_case)
        
        # 지식 규칙 추출
        if analysis_result.get("knowledge_rule"):
            rule_data = analysis_result["knowledge_rule"]
            rule = KnowledgeRule(**rule_data)
            return rule
        
        return None

