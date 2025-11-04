"""
데이터 수집 API 라우트
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
from datetime import datetime

from app.data_integration.collector import MultimodalDataCollector
from app.data_integration.preprocessor import DataPreprocessor

router = APIRouter(prefix="/api/v1/data", tags=["data-collection"])

# 전역 인스턴스
collector = MultimodalDataCollector()
preprocessor = DataPreprocessor()


@router.post("/text")
async def upload_text(
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    텍스트 데이터 업로드
    
    Args:
        student_id: 학생 ID
        file: 텍스트 파일
        
    Returns:
        업로드된 텍스트 문서 정보
    """
    try:
        file_content = await file.read()
        
        text_doc = await collector.collect_text(
            student_id=student_id,
            file_content=file_content,
            file_name=file.filename,
            mime_type=file.content_type or "text/plain"
        )
        
        return {
            "success": True,
            "document": text_doc.dict(),
            "message": "텍스트 데이터가 성공적으로 업로드되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image")
async def upload_image(
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    이미지 데이터 업로드
    
    Args:
        student_id: 학생 ID
        file: 이미지 파일
        
    Returns:
        업로드된 이미지 문서 정보
    """
    try:
        file_content = await file.read()
        
        image_doc = await collector.collect_image(
            student_id=student_id,
            file_content=file_content,
            file_name=file.filename,
            mime_type=file.content_type or "image/jpeg"
        )
        
        return {
            "success": True,
            "document": image_doc.dict(),
            "message": "이미지 데이터가 성공적으로 업로드되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio")
async def upload_audio(
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    음성 데이터 업로드
    
    Args:
        student_id: 학생 ID
        file: 음성 파일
        
    Returns:
        업로드된 음성 문서 정보
    """
    try:
        file_content = await file.read()
        
        audio_doc = await collector.collect_audio(
            student_id=student_id,
            file_content=file_content,
            file_name=file.filename,
            mime_type=file.content_type or "audio/mpeg"
        )
        
        return {
            "success": True,
            "document": audio_doc.dict(),
            "message": "음성 데이터가 성공적으로 업로드되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video")
async def upload_video(
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    비디오 데이터 업로드
    
    Args:
        student_id: 학생 ID
        file: 비디오 파일
        
    Returns:
        업로드된 비디오 문서 정보
    """
    try:
        file_content = await file.read()
        
        video_doc = await collector.collect_video(
            student_id=student_id,
            file_content=file_content,
            file_name=file.filename,
            mime_type=file.content_type or "video/mp4"
        )
        
        return {
            "success": True,
            "document": video_doc.dict(),
            "message": "비디오 데이터가 성공적으로 업로드되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preprocess")
async def preprocess_data(
    student_id: str
):
    """
    데이터 전처리
    
    Args:
        student_id: 학생 ID
        
    Returns:
        전처리된 데이터 정보
    """
    try:
        # TODO: 데이터베이스에서 MultimodalData 조회
        # 현재는 더미 데이터로 처리
        
        from app.data_integration.collector import MultimodalData
        
        multimodal_data = MultimodalData(
            student_id=student_id,
            text_data=[],
            image_data=[],
            audio_data=[],
            video_data=[]
        )
        
        preprocessed = await preprocessor.preprocess_multimodal(multimodal_data)
        
        return {
            "success": True,
            "preprocessed_data": {
                "student_id": preprocessed.student_id,
                "text_embeddings_count": len(preprocessed.text_embeddings) if preprocessed.text_embeddings else 0,
                "image_features_count": len(preprocessed.image_features) if preprocessed.image_features else 0,
                "audio_features_count": len(preprocessed.audio_features) if preprocessed.audio_features else 0,
                "video_features_count": len(preprocessed.video_features) if preprocessed.video_features else 0,
            },
            "message": "데이터 전처리가 완료되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

