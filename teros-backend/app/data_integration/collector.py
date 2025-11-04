"""
멀티모달 데이터 수집기 (Multimodal Data Collector)

학생의 학습 과정에서 발생하는 모든 정형·비정형·멀티모달 데이터를 수집
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field


class TextDocument(BaseModel):
    """텍스트 문서 데이터 구조"""
    content: str
    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    uploaded_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageDocument(BaseModel):
    """이미지 문서 데이터 구조"""
    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    uploaded_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioDocument(BaseModel):
    """음성 문서 데이터 구조"""
    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    uploaded_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VideoDocument(BaseModel):
    """비디오 문서 데이터 구조"""
    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    uploaded_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextAnnotation(BaseModel):
    """맥락 어노테이션 데이터 구조"""
    annotation_type: str  # 'teacher_tag', 'student_metacognition'
    content: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    created_by: str  # teacher_id or student_id
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalData(BaseModel):
    """멀티모달 데이터 통합 구조"""
    student_id: str
    text_data: List[TextDocument] = Field(default_factory=list)
    image_data: List[ImageDocument] = Field(default_factory=list)
    audio_data: List[AudioDocument] = Field(default_factory=list)
    video_data: List[VideoDocument] = Field(default_factory=list)
    context_annotations: List[ContextAnnotation] = Field(default_factory=list)
    collected_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalDataCollector:
    """멀티모달 데이터 수집기"""
    
    def __init__(self, storage_path: str = "./storage"):
        """
        Args:
            storage_path: 파일 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 각 데이터 타입별 저장 경로
        self.text_path = self.storage_path / "text"
        self.image_path = self.storage_path / "image"
        self.audio_path = self.storage_path / "audio"
        self.video_path = self.storage_path / "video"
        
        # 디렉토리 생성
        for path in [self.text_path, self.image_path, self.image_path, self.audio_path, self.video_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def collect_text(
        self,
        student_id: str,
        file_content: bytes,
        file_name: str,
        mime_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TextDocument:
        """
        텍스트 데이터 수집
        
        Args:
            student_id: 학생 ID
            file_content: 파일 내용 (bytes)
            file_name: 파일 이름
            mime_type: MIME 타입
            metadata: 추가 메타데이터
            
        Returns:
            TextDocument 객체
        """
        # 파일 저장
        file_path = self._save_file(self.text_path, student_id, file_name, file_content)
        
        # 텍스트 추출
        text_content = self._extract_text(file_content, mime_type)
        
        return TextDocument(
            content=text_content,
            file_path=str(file_path),
            file_name=file_name,
            file_size=len(file_content),
            mime_type=mime_type,
            uploaded_at=datetime.now(),
            metadata=metadata or {}
        )
    
    async def collect_image(
        self,
        student_id: str,
        file_content: bytes,
        file_name: str,
        mime_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ImageDocument:
        """
        이미지 데이터 수집
        
        Args:
            student_id: 학생 ID
            file_content: 파일 내용 (bytes)
            file_name: 파일 이름
            mime_type: MIME 타입
            metadata: 추가 메타데이터
            
        Returns:
            ImageDocument 객체
        """
        # 파일 저장
        file_path = self._save_file(self.image_path, student_id, file_name, file_content)
        
        # 이미지 메타데이터 추출
        width, height = self._extract_image_metadata(file_content)
        
        return ImageDocument(
            file_path=str(file_path),
            file_name=file_name,
            file_size=len(file_content),
            mime_type=mime_type,
            width=width,
            height=height,
            uploaded_at=datetime.now(),
            metadata=metadata or {}
        )
    
    async def collect_audio(
        self,
        student_id: str,
        file_content: bytes,
        file_name: str,
        mime_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AudioDocument:
        """
        음성 데이터 수집
        
        Args:
            student_id: 학생 ID
            file_content: 파일 내용 (bytes)
            file_name: 파일 이름
            mime_type: MIME 타입
            metadata: 추가 메타데이터
            
        Returns:
            AudioDocument 객체
        """
        # 파일 저장
        file_path = self._save_file(self.audio_path, student_id, file_name, file_content)
        
        # 오디오 메타데이터 추출
        duration, sample_rate = self._extract_audio_metadata(file_content)
        
        return AudioDocument(
            file_path=str(file_path),
            file_name=file_name,
            file_size=len(file_content),
            mime_type=mime_type,
            duration=duration,
            sample_rate=sample_rate,
            uploaded_at=datetime.now(),
            metadata=metadata or {}
        )
    
    async def collect_video(
        self,
        student_id: str,
        file_content: bytes,
        file_name: str,
        mime_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VideoDocument:
        """
        비디오 데이터 수집
        
        Args:
            student_id: 학생 ID
            file_content: 파일 내용 (bytes)
            file_name: 파일 이름
            mime_type: MIME 타입
            metadata: 추가 메타데이터
            
        Returns:
            VideoDocument 객체
        """
        # 파일 저장
        file_path = self._save_file(self.video_path, student_id, file_name, file_content)
        
        # 비디오 메타데이터 추출
        width, height, duration, fps = self._extract_video_metadata(file_content)
        
        return VideoDocument(
            file_path=str(file_path),
            file_name=file_name,
            file_size=len(file_content),
            mime_type=mime_type,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            uploaded_at=datetime.now(),
            metadata=metadata or {}
        )
    
    def _save_file(
        self,
        base_path: Path,
        student_id: str,
        file_name: str,
        file_content: bytes
    ) -> Path:
        """파일 저장"""
        student_dir = base_path / student_id
        student_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = student_dir / file_name
        file_path.write_bytes(file_content)
        
        return file_path
    
    def _extract_text(self, file_content: bytes, mime_type: str) -> str:
        """텍스트 추출"""
        # TODO: MIME 타입에 따라 다양한 텍스트 추출 로직 구현
        # 현재는 기본 텍스트 파일만 처리
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('utf-8', errors='ignore')
    
    def _extract_image_metadata(self, file_content: bytes) -> tuple[Optional[int], Optional[int]]:
        """이미지 메타데이터 추출 (width, height)"""
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(file_content))
            return image.size  # (width, height)
        except Exception:
            return None, None
    
    def _extract_audio_metadata(
        self,
        file_content: bytes
    ) -> tuple[Optional[float], Optional[int]]:
        """오디오 메타데이터 추출 (duration, sample_rate)"""
        try:
            import librosa
            import io
            
            y, sr = librosa.load(io.BytesIO(file_content), sr=None)
            duration = len(y) / sr if sr else None
            
            return duration, sr
        except Exception:
            return None, None
    
    def _extract_video_metadata(
        self,
        file_content: bytes
    ) -> tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        """비디오 메타데이터 추출 (width, height, duration, fps)"""
        try:
            import cv2
            import numpy as np
            import tempfile
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            cap = cv2.VideoCapture(tmp_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps else None
            
            cap.release()
            Path(tmp_path).unlink()  # 임시 파일 삭제
            
            return width, height, duration, fps
        except Exception:
            return None, None, None, None
    
    def validate_data(self, data: MultimodalData) -> bool:
        """
        수집된 데이터 검증
        
        Args:
            data: MultimodalData 객체
            
        Returns:
            검증 성공 여부
        """
        # 기본 검증
        if not data.student_id:
            return False
        
        # 파일 존재 확인
        for doc in data.text_data:
            if not Path(doc.file_path).exists():
                return False
        
        for doc in data.image_data:
            if not Path(doc.file_path).exists():
                return False
        
        for doc in data.audio_data:
            if not Path(doc.file_path).exists():
                return False
        
        for doc in data.video_data:
            if not Path(doc.file_path).exists():
                return False
        
        return True

