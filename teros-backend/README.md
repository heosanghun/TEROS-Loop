# TEROS Backend

TEROS (Teacher-Enhanced Reliability-Oriented Self-improving) 멀티모달 Agentic AI System의 백엔드 서버입니다.

## 🏗️ 프로젝트 구조

```
teros-backend/
├── app/
│   ├── data_integration/     # 데이터 통합 계층
│   ├── analytics/            # 신뢰성 기반 분석 계층
│   ├── engine/               # 자가 발전 엔진 계층
│   ├── api/                  # API 엔드포인트
│   └── models/               # 데이터 모델
├── tests/                    # 테스트 코드
├── requirements.txt          # Python 의존성
└── README.md                 # 이 파일
```

## 🚀 시작하기

### 필수 요구사항

- Python 3.10 이상
- PostgreSQL 14 이상
- MongoDB 6.0 이상
- Node.js 18 이상 (프론트엔드 개발 시)

### 설치

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 설정 추가
```

4. 데이터베이스 마이그레이션:
```bash
alembic upgrade head
```

5. 서버 실행:
```bash
uvicorn app.main:app --reload
```

## 📝 개발 가이드

### 코드 스타일

- Python 코드 스타일: Black, isort
- 타입 힌트: mypy
- 린팅: flake8

### 테스트

```bash
# 모든 테스트 실행
pytest

# 커버리지 포함
pytest --cov=app --cov-report=html
```

### API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 기술 스택

- **Web Framework**: FastAPI
- **AI/ML**: PyTorch, Transformers
- **Database**: PostgreSQL, MongoDB
- **LLM**: OpenAI GPT-4, Llama 3
- **Vision**: Vision Transformer, CNN
- **Audio**: Whisper
- **XAI**: Captum, Grad-CAM
- **Ontology**: RDFLib, OWLReady2

## 📚 참고 자료

- [TEROS 개발 계획서](../TEROS_개발계획서.md)
- [TEROS 아키텍처 설계](../TEROS_Architecture_Detail.md)

