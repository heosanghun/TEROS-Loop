"""
TEROS Backend Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import data_collection, analytics, teros_loop

app = FastAPI(
    title="TEROS API",
    description="TEROS 멀티모달 Agentic AI System API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우트 등록
app.include_router(data_collection.router)
app.include_router(analytics.router)
app.include_router(teros_loop.router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "TEROS API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}

