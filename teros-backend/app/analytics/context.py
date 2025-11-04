"""
맥락 이해 강화 모듈 (Context Awareness Module)

교사-학생 참여형 어노테이션 시스템
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer


class ContextTag(BaseModel):
    """컨텍스트 태그"""
    tag_id: str
    tag_name: str
    tag_category: str  # 'condition', 'relationship', 'academic', 'personal'
    description: Optional[str] = None


class TeacherAnnotation(BaseModel):
    """교사 어노테이션"""
    annotation_id: str
    student_id: str
    teacher_id: str
    document_id: str  # 어노테이션 대상 문서 ID
    document_type: str  # 'text', 'image', 'audio', 'video'
    tags: List[str] = Field(default_factory=list)  # 태그 리스트
    free_text: Optional[str] = None  # 자유 텍스트 주석
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudentMetacognitionReport(BaseModel):
    """학생 메타인지 리포트"""
    report_id: str
    student_id: str
    assignment_id: str
    content: str  # 메타인지 노트 내용
    difficulty_points: List[str] = Field(default_factory=list)  # 어려웠던 점
    learned_points: List[str] = Field(default_factory=list)  # 새로 배운 점
    reflection: Optional[str] = None  # 성찰 내용
    created_at: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None  # BERT 임베딩
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextAwarenessModule:
    """맥락 이해 강화 모듈"""
    
    # 기본 태그 정의
    DEFAULT_TAGS = [
        # 컨디션 관련
        ContextTag(tag_id="cond_bad", tag_name="컨디션난조", tag_category="condition"),
        ContextTag(tag_id="cond_good", tag_name="컨디션양호", tag_category="condition"),
        ContextTag(tag_id="cond_stress", tag_name="스트레스", tag_category="condition"),
        
        # 관계 관련
        ContextTag(tag_id="rel_peer_conflict", tag_name="또래갈등", tag_category="relationship"),
        ContextTag(tag_id="rel_peer_cooperation", tag_name="또래협력", tag_category="relationship"),
        ContextTag(tag_id="rel_teacher_support", tag_name="교사지원", tag_category="relationship"),
        
        # 학업 관련
        ContextTag(tag_id="acad_high_interest", tag_name="높은흥미", tag_category="academic"),
        ContextTag(tag_id="acad_low_interest", tag_name="낮은흥미", tag_category="academic"),
        ContextTag(tag_id="acad_understanding", tag_name="이해도높음", tag_category="academic"),
        ContextTag(tag_id="acad_confusion", tag_name="혼란", tag_category="academic"),
        
        # 개인 관련
        ContextTag(tag_id="pers_stage_fright", tag_name="극심한무대공포증", tag_category="personal"),
        ContextTag(tag_id="pers_family_issue", tag_name="가정사", tag_category="personal"),
        ContextTag(tag_id="pers_health", tag_name="건강문제", tag_category="personal"),
    ]
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            embedding_model_name: BERT 임베딩 모델 이름
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.tags = {tag.tag_id: tag for tag in self.DEFAULT_TAGS}
        self.annotations: Dict[str, TeacherAnnotation] = {}
        self.metacognition_reports: Dict[str, StudentMetacognitionReport] = {}
    
    def get_tags(self, category: Optional[str] = None) -> List[ContextTag]:
        """
        태그 목록 조회
        
        Args:
            category: 태그 카테고리 필터링
            
        Returns:
            태그 리스트
        """
        if category:
            return [tag for tag in self.tags.values() if tag.tag_category == category]
        return list(self.tags.values())
    
    def add_teacher_annotation(
        self,
        annotation: TeacherAnnotation
    ) -> TeacherAnnotation:
        """
        교사 어노테이션 추가
        
        Args:
            annotation: 교사 어노테이션 객체
            
        Returns:
            추가된 어노테이션
        """
        # 태그 검증
        for tag_id in annotation.tags:
            if tag_id not in self.tags:
                raise ValueError(f"Unknown tag: {tag_id}")
        
        self.annotations[annotation.annotation_id] = annotation
        return annotation
    
    def add_metacognition_report(
        self,
        report: StudentMetacognitionReport
    ) -> StudentMetacognitionReport:
        """
        학생 메타인지 리포트 추가
        
        Args:
            report: 메타인지 리포트 객체
            
        Returns:
            추가된 리포트 (임베딩 포함)
        """
        # 메타인지 내용 임베딩 생성
        metacognition_text = self._construct_metacognition_text(report)
        embedding = self.embedding_model.encode(metacognition_text)
        report.embedding = embedding.tolist()
        
        self.metacognition_reports[report.report_id] = report
        return report
    
    def get_context_for_student(
        self,
        student_id: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        학생의 맥락 정보 조회
        
        Args:
            student_id: 학생 ID
            document_id: 특정 문서 ID (선택사항)
            
        Returns:
            맥락 정보
        """
        # 교사 어노테이션 조회
        annotations = [
            ann for ann in self.annotations.values()
            if ann.student_id == student_id
            and (document_id is None or ann.document_id == document_id)
        ]
        
        # 메타인지 리포트 조회
        reports = [
            report for report in self.metacognition_reports.values()
            if report.student_id == student_id
        ]
        
        return {
            "student_id": student_id,
            "annotations": [ann.dict() for ann in annotations],
            "metacognition_reports": [report.dict() for report in reports],
            "context_summary": self._generate_context_summary(annotations, reports)
        }
    
    def integrate_context_to_features(
        self,
        base_features: np.ndarray,
        student_id: str,
        document_id: Optional[str] = None
    ) -> np.ndarray:
        """
        맥락 정보를 특징 벡터에 통합
        
        Args:
            base_features: 기본 특징 벡터
            student_id: 학생 ID
            document_id: 문서 ID
            
        Returns:
            맥락 정보가 통합된 특징 벡터
        """
        context = self.get_context_for_student(student_id, document_id)
        
        # 태그 원-핫 인코딩
        tag_vector = self._encode_tags(context["annotations"])
        
        # 메타인지 임베딩 평균
        metacognition_vector = self._get_metacognition_embedding(
            context["metacognition_reports"]
        )
        
        # 특징 벡터 통합
        if metacognition_vector is not None:
            enhanced_features = np.concatenate([
                base_features,
                tag_vector,
                metacognition_vector
            ])
        else:
            enhanced_features = np.concatenate([
                base_features,
                tag_vector
            ])
        
        return enhanced_features
    
    def _construct_metacognition_text(self, report: StudentMetacognitionReport) -> str:
        """메타인지 텍스트 구성"""
        parts = []
        
        if report.difficulty_points:
            parts.append(f"어려웠던 점: {', '.join(report.difficulty_points)}")
        
        if report.learned_points:
            parts.append(f"새로 배운 점: {', '.join(report.learned_points)}")
        
        if report.reflection:
            parts.append(f"성찰: {report.reflection}")
        
        if report.content:
            parts.append(report.content)
        
        return " ".join(parts)
    
    def _generate_context_summary(
        self,
        annotations: List[TeacherAnnotation],
        reports: List[StudentMetacognitionReport]
    ) -> Dict[str, Any]:
        """맥락 요약 생성"""
        # 태그 빈도 계산
        tag_counts = {}
        for ann in annotations:
            for tag_id in ann.tags:
                tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
        
        # 주요 태그 추출
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_annotations": len(annotations),
            "total_metacognition_reports": len(reports),
            "top_tags": [{"tag_id": tag_id, "count": count} for tag_id, count in top_tags],
            "has_personal_context": any(
                any(tag_id.startswith("pers_") for tag_id in ann.tags)
                for ann in annotations
            ),
            "has_academic_context": any(
                any(tag_id.startswith("acad_") for tag_id in ann.tags)
                for ann in annotations
            )
        }
    
    def _encode_tags(self, annotations: List[Dict[str, Any]]) -> np.ndarray:
        """태그 원-핫 인코딩"""
        tag_vector = np.zeros(len(self.tags))
        tag_ids = list(self.tags.keys())
        
        for ann in annotations:
            for tag_id in ann.get("tags", []):
                if tag_id in tag_ids:
                    tag_vector[tag_ids.index(tag_id)] = 1.0
        
        return tag_vector
    
    def _get_metacognition_embedding(
        self,
        reports: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """메타인지 임베딩 평균"""
        if not reports:
            return None
        
        embeddings = []
        for report in reports:
            if report.get("embedding"):
                embeddings.append(report["embedding"])
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def adjust_context_weights(
        self,
        features: np.ndarray,
        context_importance: float = 0.3
    ) -> np.ndarray:
        """
        맥락 가중치 조정
        
        Args:
            features: 특징 벡터
            context_importance: 맥락 중요도 (0-1)
            
        Returns:
            가중치 조정된 특징 벡터
        """
        # 기본 특징과 맥락 특징 분리 (가정: 맥락 특징이 뒤에 추가됨)
        base_dim = int(len(features) * (1 - context_importance))
        base_features = features[:base_dim]
        context_features = features[base_dim:]
        
        # 맥락 특징에 가중치 적용
        weighted_context = context_features * context_importance
        
        # 통합
        return np.concatenate([base_features, weighted_context])

