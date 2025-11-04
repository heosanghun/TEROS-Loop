"""
Pytest 설정 및 픽스처
"""
import pytest
import os
from pathlib import Path

# 테스트용 임시 저장소 경로
TEST_STORAGE_PATH = "./test_storage"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """테스트 환경 설정"""
    # 테스트 저장소 디렉토리 생성
    Path(TEST_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # 테스트 후 정리
    import shutil
    if Path(TEST_STORAGE_PATH).exists():
        shutil.rmtree(TEST_STORAGE_PATH)


@pytest.fixture
def mock_multimodal_data():
    """멀티모달 데이터 모킹"""
    return {
        "student_id": "test_student_001",
        "text_data": [],
        "image_data": [],
        "audio_data": [],
        "video_data": [],
        "context_annotations": []
    }

