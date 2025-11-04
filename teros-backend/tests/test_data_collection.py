"""
데이터 수집 모듈 테스트
"""
import pytest
from pathlib import Path
from datetime import datetime

from app.data_integration.collector import (
    MultimodalDataCollector,
    TextDocument,
    ImageDocument,
    AudioDocument,
    VideoDocument,
    MultimodalData
)


@pytest.fixture
def collector():
    """MultimodalDataCollector 인스턴스 생성"""
    return MultimodalDataCollector(storage_path="./test_storage")


@pytest.fixture
def sample_text_content():
    """샘플 텍스트 내용"""
    return b"This is a sample text document for testing."


@pytest.fixture
def sample_image_content():
    """샘플 이미지 내용 (더미)"""
    # 실제 테스트에서는 실제 이미지 파일 사용
    return b"dummy image content"


@pytest.mark.asyncio
async def test_collect_text(collector, sample_text_content):
    """텍스트 데이터 수집 테스트"""
    text_doc = await collector.collect_text(
        student_id="test_student_001",
        file_content=sample_text_content,
        file_name="test.txt",
        mime_type="text/plain"
    )
    
    assert isinstance(text_doc, TextDocument)
    assert text_doc.file_name == "test.txt"
    assert text_doc.content == sample_text_content.decode('utf-8')
    assert text_doc.file_size == len(sample_text_content)
    assert Path(text_doc.file_path).exists()


@pytest.mark.asyncio
async def test_collect_image(collector, sample_image_content):
    """이미지 데이터 수집 테스트"""
    image_doc = await collector.collect_image(
        student_id="test_student_001",
        file_content=sample_image_content,
        file_name="test.jpg",
        mime_type="image/jpeg"
    )
    
    assert isinstance(image_doc, ImageDocument)
    assert image_doc.file_name == "test.jpg"
    assert image_doc.file_size == len(sample_image_content)
    assert Path(image_doc.file_path).exists()


def test_validate_data(collector):
    """데이터 검증 테스트"""
    # 유효한 데이터
    valid_data = MultimodalData(
        student_id="test_student_001",
        text_data=[],
        image_data=[],
        audio_data=[],
        video_data=[]
    )
    
    assert collector.validate_data(valid_data) is True
    
    # 유효하지 않은 데이터 (student_id 없음)
    invalid_data = MultimodalData(
        student_id="",
        text_data=[],
        image_data=[],
        audio_data=[],
        video_data=[]
    )
    
    assert collector.validate_data(invalid_data) is False


def test_multimodal_data_structure():
    """MultimodalData 구조 테스트"""
    data = MultimodalData(
        student_id="test_student_001",
        text_data=[],
        image_data=[],
        audio_data=[],
        video_data=[],
        context_annotations=[]
    )
    
    assert data.student_id == "test_student_001"
    assert isinstance(data.collected_at, datetime)
    assert isinstance(data.metadata, dict)

