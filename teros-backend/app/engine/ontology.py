"""
교육 지식 온톨로지 (Educational Knowledge Ontology)

OWL 기반 지식 베이스 및 점진적 확장
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL
from owlready2 import *
import uuid


class EducationalKnowledgeOntology:
    """교육 지식 온톨로지"""
    
    # 기본 네임스페이스
    TEROS_NS = Namespace("http://teros.edu/ontology/")
    
    def __init__(self, ontology_path: str = "./ontology/teros_ontology.owl"):
        """
        Args:
            ontology_path: 온톨로지 파일 경로
        """
        self.ontology_path = ontology_path
        self.graph = Graph()
        self.graph.bind("teros", self.TEROS_NS)
        
        # 핵심 클래스 정의
        self._initialize_core_classes()
    
    def _initialize_core_classes(self):
        """핵심 클래스 초기화"""
        # 핵심 클래스 정의
        Student = self.TEROS_NS.Student
        Talent = self.TEROS_NS.Talent
        LearningActivity = self.TEROS_NS.LearningActivity
        Career = self.TEROS_NS.Career
        KnowledgeRule = self.TEROS_NS.KnowledgeRule
        
        # 관계 정의
        hasTalent = self.TEROS_NS.hasTalent
        participatesIn = self.TEROS_NS.participatesIn
        recommendsCareer = self.TEROS_NS.recommendsCareer
        appliesRule = self.TEROS_NS.appliesRule
        
        # 클래스 정의를 그래프에 추가
        self.graph.add((Student, RDF.type, OWL.Class))
        self.graph.add((Talent, RDF.type, OWL.Class))
        self.graph.add((LearningActivity, RDF.type, OWL.Class))
        self.graph.add((Career, RDF.type, OWL.Class))
        self.graph.add((KnowledgeRule, RDF.type, OWL.Class))
        
        # 관계 정의
        self.graph.add((hasTalent, RDF.type, OWL.ObjectProperty))
        self.graph.add((participatesIn, RDF.type, OWL.ObjectProperty))
        self.graph.add((recommendsCareer, RDF.type, OWL.ObjectProperty))
        self.graph.add((appliesRule, RDF.type, OWL.ObjectProperty))
    
    def integrate_knowledge_rule(
        self,
        rule: Dict[str, Any],
        rule_id: Optional[str] = None
    ) -> str:
        """
        지식 규칙을 온톨로지에 통합
        
        Args:
            rule: 지식 규칙 (IF-THEN 형태)
            rule_id: 규칙 ID (없으면 자동 생성)
            
        Returns:
            규칙 ID
        """
        if rule_id is None:
            rule_id = f"rule_{uuid.uuid4().hex[:16]}"
        
        rule_uri = self.TEROS_NS[rule_id]
        KnowledgeRule = self.TEROS_NS.KnowledgeRule
        
        # 규칙을 클래스로 추가
        self.graph.add((rule_uri, RDF.type, KnowledgeRule))
        
        # 조건과 결과를 속성으로 추가
        condition = rule.get("condition", {})
        result = rule.get("result", {})
        
        # 조건을 하위 클래스나 관계로 변환
        if "type" in condition:
            condition_class = self.TEROS_NS[condition["type"]]
            self.graph.add((condition_class, RDF.type, OWL.Class))
            self.graph.add((condition_class, RDFS.subClassOf, LearningActivity))
            self.graph.add((rule_uri, self.TEROS_NS.hasCondition, condition_class))
        
        # 결과를 하위 클래스나 관계로 변환
        if "type" in result:
            result_class = self.TEROS_NS[result["type"]]
            self.graph.add((result_class, RDF.type, OWL.Class))
            self.graph.add((result_class, RDFS.subClassOf, Talent))
            self.graph.add((rule_uri, self.TEROS_NS.hasResult, result_class))
        
        # 온톨로지 저장
        self.save_ontology()
        
        return rule_id
    
    def add_talent_class(
        self,
        talent_id: str,
        talent_name: str,
        parent_talent: Optional[str] = None
    ):
        """
        재능 클래스 추가
        
        Args:
            talent_id: 재능 ID
            talent_name: 재능 이름
            parent_talent: 부모 재능 (선택사항)
        """
        talent_uri = self.TEROS_NS[talent_id]
        Talent = self.TEROS_NS.Talent
        
        # 클래스 정의
        self.graph.add((talent_uri, RDF.type, OWL.Class))
        self.graph.add((talent_uri, RDFS.subClassOf, Talent))
        self.graph.add((talent_uri, RDFS.label, Literal(talent_name, lang="ko")))
        
        # 부모 재능이 있으면 하위 클래스로 설정
        if parent_talent:
            parent_uri = self.TEROS_NS[parent_talent]
            self.graph.add((talent_uri, RDFS.subClassOf, parent_uri))
        
        self.save_ontology()
    
    def add_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str
    ):
        """
        관계 추가
        
        Args:
            subject_id: 주체 ID
            predicate: 술어 (관계)
            object_id: 객체 ID
        """
        subject_uri = self.TEROS_NS[subject_id]
        object_uri = self.TEROS_NS[object_id]
        predicate_uri = self.TEROS_NS[predicate]
        
        self.graph.add((subject_uri, predicate_uri, object_uri))
        self.save_ontology()
    
    def query_ontology(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        온톨로지 쿼리 (SPARQL)
        
        Args:
            query: SPARQL 쿼리
            
        Returns:
            쿼리 결과
        """
        results = self.graph.query(query)
        return [dict(row) for row in results]
    
    def save_ontology(self):
        """온톨로지 저장"""
        import os
        os.makedirs(os.path.dirname(self.ontology_path), exist_ok=True)
        self.graph.serialize(destination=self.ontology_path, format="xml")
    
    def load_ontology(self):
        """온톨로지 로드"""
        import os
        if os.path.exists(self.ontology_path):
            self.graph.parse(self.ontology_path, format="xml")
    
    def get_all_talents(self) -> List[Dict[str, Any]]:
        """모든 재능 클래스 조회"""
        query = """
        PREFIX teros: <http://teros.edu/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?talent ?label
        WHERE {
            ?talent rdfs:subClassOf teros:Talent .
            ?talent rdfs:label ?label .
        }
        """
        
        results = self.query_ontology(query)
        return [
            {
                "id": str(r["talent"]).split("/")[-1],
                "name": str(r["label"])
            }
            for r in results
        ]
    
    def get_knowledge_rules(self) -> List[Dict[str, Any]]:
        """모든 지식 규칙 조회"""
        query = """
        PREFIX teros: <http://teros.edu/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?rule ?condition ?result
        WHERE {
            ?rule rdf:type teros:KnowledgeRule .
            ?rule teros:hasCondition ?condition .
            ?rule teros:hasResult ?result .
        }
        """
        
        results = self.query_ontology(query)
        return [
            {
                "rule_id": str(r["rule"]).split("/")[-1],
                "condition": str(r["condition"]).split("/")[-1],
                "result": str(r["result"]).split("/")[-1]
            }
            for r in results
        ]

