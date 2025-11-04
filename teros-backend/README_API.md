# TEROS API 문서

## API 엔드포인트

### 데이터 수집 API

#### POST `/api/v1/data/text`
텍스트 데이터 업로드

**요청:**
- Form Data:
  - `student_id`: 학생 ID
  - `file`: 텍스트 파일

**응답:**
```json
{
  "success": true,
  "document": {
    "content": "...",
    "file_path": "...",
    "file_name": "...",
    "file_size": 1234,
    "mime_type": "text/plain",
    "uploaded_at": "2024-01-01T00:00:00"
  },
  "message": "텍스트 데이터가 성공적으로 업로드되었습니다."
}
```

#### POST `/api/v1/data/image`
이미지 데이터 업로드

#### POST `/api/v1/data/audio`
음성 데이터 업로드

#### POST `/api/v1/data/video`
비디오 데이터 업로드

#### POST `/api/v1/data/preprocess`
데이터 전처리

### 분석 API

#### POST `/api/v1/analytics/talent-diagnosis`
재능 진단 분석

**요청:**
```json
{
  "student_id": "student_001",
  "text_features": [[0.1, 0.2, ...]],
  "image_features": [[0.3, 0.4, ...]],
  "audio_features": [[0.5, 0.6, ...]],
  "video_features": [[0.7, 0.8, ...]],
  "base_features": [0.9, 0.10, ...]
}
```

**응답:**
```json
{
  "success": true,
  "profile": {
    "student_id": "student_001",
    "talents": [...],
    "overall_score": 75.5,
    "top_talents": ["logical", "creative"],
    "career_recommendations": ["프로그래머", "디자이너"],
    "learning_path": [...]
  }
}
```

#### POST `/api/v1/analytics/explain`
설명 생성

#### POST `/api/v1/analytics/counterfactual`
반사실적 설명 생성

#### POST `/api/v1/analytics/context/annotation`
교사 어노테이션 추가

#### POST `/api/v1/analytics/context/metacognition`
메타인지 리포트 추가

#### GET `/api/v1/analytics/context/{student_id}`
맥락 정보 조회

#### GET `/api/v1/analytics/context/tags`
태그 목록 조회

#### POST `/api/v1/analytics/fairness/evaluate`
공정성 평가

### TEROS-Loop API

#### POST `/api/v1/teros-loop/process-discrepancy`
불일치 사례 처리

#### POST `/api/v1/teros-loop/update-confidence`
규칙 신뢰도 업데이트

#### GET `/api/v1/teros-loop/deliberation/pending`
대기 중인 공동 심의 안건 조회

#### POST `/api/v1/teros-loop/deliberation/respond`
교사 응답 제출

#### GET `/api/v1/teros-loop/statistics`
TEROS-Loop 통계

#### GET `/api/v1/teros-loop/cases/pending`
대기 중인 불일치 사례 조회

## Swagger UI

서버 실행 후 다음 URL에서 상세 API 문서를 확인할 수 있습니다:
- http://localhost:8000/docs

