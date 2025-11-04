# TEROS 배포 가이드

## 배포 개요

TEROS 시스템을 프로덕션 환경에 배포하는 방법을 안내합니다.

## 사전 요구사항

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- MongoDB 6.0+
- Docker 20.10+ (선택사항)

## 배포 방법

### 방법 1: Docker Compose 사용 (권장)

1. **환경 변수 설정**
```bash
cp teros-backend/.env.example teros-backend/.env
cp teros-frontend/.env.example teros-frontend/.env
# .env 파일 편집
```

2. **Docker Compose로 실행**
```bash
docker-compose up -d
```

3. **서비스 확인**
- 백엔드: http://localhost:8000
- 프론트엔드: http://localhost:3000
- PostgreSQL: localhost:5432
- MongoDB: localhost:27017

### 방법 2: 수동 배포

#### 백엔드 배포

1. **가상환경 설정**
```bash
cd teros-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일 편집
```

3. **데이터베이스 마이그레이션**
```bash
alembic upgrade head
```

4. **서버 실행**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 프론트엔드 배포

1. **의존성 설치**
```bash
cd teros-frontend
npm install
```

2. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일 편집
```

3. **빌드**
```bash
npm run build
```

4. **프로덕션 서버 실행**
```bash
npm run preview
```

## 프로덕션 환경 설정

### Nginx 설정 (선택사항)

```nginx
server {
    listen 80;
    server_name teros.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 환경 변수

#### 백엔드 (.env)
```env
DATABASE_URL=postgresql://user:password@localhost:5432/teros_db
MONGODB_URL=mongodb://user:password@localhost:27017/teros_buffer?authSource=admin
OPENAI_API_KEY=your_openai_api_key
SECRET_KEY=your_secret_key
DEBUG=False
```

#### 프론트엔드 (.env)
```env
VITE_API_URL=https://api.teros.example.com
VITE_ENV=production
```

## 모니터링

### 로그 확인

```bash
# Docker Compose 로그
docker-compose logs -f

# 백엔드 로그
tail -f teros-backend/logs/app.log

# 프론트엔드 로그
tail -f teros-frontend/logs/app.log
```

### 헬스 체크

```bash
curl http://localhost:8000/health
```

## 백업 및 복구

### 데이터베이스 백업

```bash
# PostgreSQL 백업
pg_dump -U teros teros_db > backup_$(date +%Y%m%d).sql

# MongoDB 백업
mongodump --uri="mongodb://user:password@localhost:27017/teros_buffer"
```

### 복구

```bash
# PostgreSQL 복구
psql -U teros teros_db < backup_20240101.sql

# MongoDB 복구
mongorestore --uri="mongodb://user:password@localhost:27017/teros_buffer" dump/
```

## 보안 체크리스트

- [ ] 환경 변수에 민감한 정보 저장
- [ ] HTTPS 사용 (프로덕션)
- [ ] API 키 안전하게 관리
- [ ] 데이터베이스 접근 제한
- [ ] 방화벽 설정
- [ ] 정기적 보안 업데이트

## 문제 해결

### 일반적인 문제

1. **포트 충돌**
   - 포트가 이미 사용 중인 경우 다른 포트 사용

2. **데이터베이스 연결 실패**
   - 연결 문자열 확인
   - 데이터베이스 서비스 실행 확인

3. **권한 문제**
   - 파일 및 디렉토리 권한 확인

## 지원

문제가 발생하면 GitHub Issues에 문의하세요.

