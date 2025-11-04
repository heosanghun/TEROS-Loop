# TEROS 개발 가이드라인

## 📋 목차

1. [개발 환경 설정](#개발-환경-설정)
2. [프로젝트 구조](#프로젝트-구조)
3. [코딩 규칙](#코딩-규칙)
4. [Git 워크플로우](#git-워크플로우)
5. [테스트](#테스트)
6. [문서화](#문서화)

## 🚀 개발 환경 설정

### 필수 요구사항

- **Python**: 3.10 이상
- **Node.js**: 18 이상
- **PostgreSQL**: 14 이상
- **MongoDB**: 6.0 이상
- **Docker**: 20.10 이상 (선택사항)

### 초기 설정

1. **저장소 클론**:
```bash
git clone <repository-url>
cd TEROS
```

2. **백엔드 설정**:
```bash
cd teros-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# .env 파일 편집
```

3. **프론트엔드 설정**:
```bash
cd teros-frontend
npm install
cp .env.example .env
# .env 파일 편집
```

4. **Docker로 실행** (선택사항):
```bash
docker-compose up -d
```

## 🏗️ 프로젝트 구조

### 백엔드 구조

```
teros-backend/
├── app/
│   ├── data_integration/     # 데이터 통합 계층
│   │   ├── __init__.py
│   │   ├── collector.py       # 멀티모달 데이터 수집기
│   │   └── preprocessor.py    # 데이터 전처리
│   ├── analytics/            # 신뢰성 기반 분석 계층
│   │   ├── __init__.py
│   │   ├── fairness.py        # 공정성 강화 모듈
│   │   ├── explainability.py  # 설명가능성 모듈
│   │   └── context.py         # 맥락 이해 모듈
│   ├── engine/               # 자가 발전 엔진 계층
│   │   ├── __init__.py
│   │   ├── teros_loop.py      # TEROS-Loop
│   │   └── ontology.py        # 교육 지식 온톨로지
│   ├── api/                  # API 엔드포인트
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI 앱
│   │   └── routes/            # API 라우트
│   └── models/               # 데이터 모델
│       ├── __init__.py
│       └── schemas.py        # Pydantic 스키마
├── tests/                    # 테스트 코드
├── requirements.txt
└── README.md
```

### 프론트엔드 구조

```
teros-frontend/
├── src/
│   ├── components/
│   │   ├── TeacherDashboard/
│   │   ├── StudentDashboard/
│   │   └── Deliberation/
│   ├── services/             # API 서비스
│   ├── utils/                # 유틸리티
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── README.md
```

## 📝 코딩 규칙

### Python (백엔드)

1. **코드 스타일**:
   - Black 사용 (줄 길이 100)
   - isort 사용 (import 정렬)
   - PEP 8 준수

2. **타입 힌트**:
   - 모든 함수에 타입 힌트 필수
   - mypy 검증 통과 필수

3. **Docstring**:
   - Google 스타일 사용
   - 모든 공개 함수/클래스에 문서화

4. **예시**:
```python
from typing import List, Optional

def process_data(
    data: List[str],
    threshold: Optional[float] = None
) -> List[dict]:
    """
    데이터를 처리합니다.

    Args:
        data: 처리할 데이터 리스트
        threshold: 임계값 (선택사항)

    Returns:
        처리된 데이터 리스트

    Raises:
        ValueError: 데이터가 비어있는 경우
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # 처리 로직
    return processed_data
```

### TypeScript (프론트엔드)

1. **코드 스타일**:
   - Prettier 사용
   - ESLint 준수
   - 함수형 컴포넌트 사용

2. **타입 정의**:
   - 모든 함수/변수에 타입 정의
   - 인터페이스 사용

3. **예시**:
```typescript
interface User {
  id: string;
  name: string;
  email: string;
}

const UserProfile: React.FC<{ user: User }> = ({ user }) => {
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
};
```

## 🔄 Git 워크플로우

### 브랜치 전략

- **main**: 프로덕션 브랜치
- **develop**: 개발 브랜치
- **feature/***: 기능 개발 브랜치
- **hotfix/***: 긴급 수정 브랜치

### 커밋 메시지 규칙

Conventional Commits 규칙 사용:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**타입**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 스타일 변경
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/설정 변경

**예시**:
```
feat(analytics): 공정성 강화 모듈 구현

적대적 학습 기반 편향 제거 네트워크를 구현했습니다.
- 예측 모델과 적대 모델 동시 학습
- 인과 추론 기반 잠재력 추정

Closes #123
```

### Pull Request 규칙

1. **제목**: 명확하고 간결하게
2. **설명**: 변경 사항, 테스트 방법, 관련 이슈
3. **리뷰**: 최소 1명 이상의 승인 필요
4. **테스트**: 모든 테스트 통과 필수

## 🧪 테스트

### 백엔드 테스트

```bash
# 모든 테스트 실행
pytest

# 커버리지 포함
pytest --cov=app --cov-report=html

# 특정 테스트만 실행
pytest tests/test_analytics.py
```

### 프론트엔드 테스트

```bash
# 테스트 실행
npm test

# 커버리지 포함
npm test -- --coverage
```

### 테스트 커버리지 목표

- **최소**: 70%
- **권장**: 80% 이상

## 📚 문서화

### 코드 문서화

1. **Python**: Google 스타일 Docstring
2. **TypeScript**: JSDoc 주석

### API 문서

- Swagger UI: http://localhost:8000/docs
- 자동 생성 (FastAPI)

### 프로젝트 문서

- README.md: 프로젝트 개요
- DEVELOPMENT_GUIDE.md: 개발 가이드
- ARCHITECTURE.md: 아키텍처 문서

## ✅ 체크리스트

### 커밋 전

- [ ] 코드 스타일 검사 통과 (Black, isort, ESLint)
- [ ] 타입 검사 통과 (mypy, TypeScript)
- [ ] 모든 테스트 통과
- [ ] 문서화 완료
- [ ] 코드 리뷰 요청

### PR 전

- [ ] 모든 체크리스트 완료
- [ ] PR 설명 작성
- [ ] 관련 이슈 연결
- [ ] 스크린샷/데모 (UI 변경 시)

## 📞 문의

문제가 발생하면 GitHub Issues에 문의하세요.

---

**본 가이드라인은 프로젝트 진행에 따라 업데이트됩니다.**

