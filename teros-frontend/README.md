# TEROS Frontend

TEROS (Teacher-Enhanced Reliability-Oriented Self-improving) 멀티모달 Agentic AI System의 프론트엔드 애플리케이션입니다.

## 🏗️ 프로젝트 구조

```
teros-frontend/
├── src/
│   ├── components/
│   │   ├── TeacherDashboard/    # 교사 대시보드
│   │   ├── StudentDashboard/    # 학생 대시보드
│   │   └── Deliberation/        # 공동 심의 인터페이스
│   ├── services/                # API 서비스
│   └── utils/                   # 유틸리티 함수
├── package.json
└── README.md
```

## 🚀 시작하기

### 필수 요구사항

- Node.js 18 이상
- npm 또는 yarn

### 설치

1. 의존성 설치:
```bash
npm install
# 또는
yarn install
```

2. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 API 엔드포인트 설정
```

3. 개발 서버 실행:
```bash
npm run dev
# 또는
yarn dev
```

4. 빌드:
```bash
npm run build
# 또는
yarn build
```

## 📝 개발 가이드

### 코드 스타일

- TypeScript 사용
- ESLint, Prettier 설정
- React Hooks 사용

### 컴포넌트 구조

- 함수형 컴포넌트 사용
- TypeScript 타입 정의 필수
- 재사용 가능한 컴포넌트 설계

### 상태 관리

- Zustand 사용 (전역 상태)
- React Query 사용 (서버 상태)

## 🔧 기술 스택

- **Framework**: React 18
- **Language**: TypeScript
- **Build Tool**: Vite
- **UI Library**: Material-UI
- **State Management**: Zustand
- **Data Fetching**: React Query
- **Visualization**: D3.js, Chart.js, Recharts
- **Routing**: React Router

## 📚 참고 자료

- [TEROS 개발 계획서](../TEROS_개발계획서.md)
- [TEROS 아키텍처 설계](../TEROS_Architecture_Detail.md)

