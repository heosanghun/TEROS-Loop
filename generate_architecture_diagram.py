#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEROS 아키텍처 다이어그램 생성 스크립트
"""

import os
try:
    from diagrams import Diagram, Cluster, Edge
    from diagrams.onprem.client import Users
    from diagrams.onprem.compute import Server
    from diagrams.onprem.database import PostgreSQL, MongoDB
    from diagrams.onprem.inmemory import Redis
    from diagrams.onprem.monitoring import Prometheus
    from diagrams.programming.language import Python
    from diagrams.programming.framework import React
    from diagrams.generic.storage import Storage
    from diagrams.aws.analytics import Analytics
    from diagrams.aws.ml import Sagemaker
    from diagrams.custom import Custom
except ImportError:
    print("diagrams 라이브러리 설치 중...")
    os.system(f"python -m pip install diagrams")
    from diagrams import Diagram, Cluster, Edge
    from diagrams.onprem.client import Users
    from diagrams.onprem.compute import Server
    from diagrams.onprem.database import PostgreSQL, MongoDB
    from diagrams.onprem.inmemory import Redis
    from diagrams.onprem.monitoring import Prometheus
    from diagrams.programming.language import Python
    from diagrams.programming.framework import React
    from diagrams.generic.storage import Storage
    from diagrams.aws.analytics import Analytics
    from diagrams.aws.ml import Sagemaker
    from diagrams.custom import Custom

def create_architecture_diagram():
    """TEROS 아키텍처 다이어그램 생성"""
    
    with Diagram("TEROS 멀티모달 Agentic AI System 아키텍처", 
                 filename="TEROS_Architecture", 
                 show=False,
                 direction="TB"):
        
        # 사용자 계층
        with Cluster("사용자 상호작용 계층 (User Interaction Layer)"):
            teacher = Users("교사\nTeacher")
            student = Users("학생\nStudent")
            ui_dashboard = React("대시보드\nDashboard")
            deliberation = React("공동 심의\nDeliberation")
        
        # 자가 발전 엔진 계층
        with Cluster("자가 발전 엔진 계층 (Self-Evolving Engine Layer)"):
            teros_loop = Server("TEROS-Loop\nSelf-Evolving Engine")
            
            with Cluster("TEROS-Loop 3단계"):
                detect = Analytics("1. 불일치 포착\nDiscrepancy Detection")
                analyze = Sagemaker("2. 원인 분석\nLLM Ensemble")
                validate = Analytics("3. 검증 메커니즘\nReliability Validation")
            
            buffer = MongoDB("개념적 버퍼\nConceptual Buffer")
            ontology = Storage("교육 지식 온톨로지\nKnowledge Ontology")
        
        # 신뢰성 기반 분석 계층
        with Cluster("신뢰성 기반 분석 계층 (Trustworthy Analytics Layer)"):
            with Cluster("기반 모듈"):
                fairness = Sagemaker("공정성 강화\nFairness Module")
                xai = Sagemaker("설명가능성\nXAI Module")
                context = Sagemaker("맥락 이해\nContext Module")
            
            analyzer = Sagemaker("재능 진단 분석기\nTalent Diagnosis")
        
        # 데이터 통합 계층
        with Cluster("데이터 통합 계층 (Data Integration Layer)"):
            collector = Server("멀티모달\n데이터 수집기")
            preprocessor = Analytics("데이터 전처리\nPreprocessing")
            
            text_data = Storage("텍스트\nText")
            image_data = Storage("이미지\nImage")
            audio_data = Storage("음성\nAudio")
            video_data = Storage("비디오\nVideo")
        
        # 연결 관계
        teacher >> ui_dashboard
        student >> ui_dashboard
        teacher >> deliberation
        
        ui_dashboard >> analyzer
        deliberation >> ontology
        
        text_data >> collector
        image_data >> collector
        audio_data >> collector
        video_data >> collector
        collector >> preprocessor
        preprocessor >> analyzer
        
        analyzer >> fairness
        analyzer >> xai
        analyzer >> context
        fairness >> analyzer
        xai >> analyzer
        context >> analyzer
        
        analyzer >> teros_loop
        teros_loop >> detect
        detect >> analyze
        analyze >> validate
        validate >> buffer
        buffer >> ontology
        ontology >> analyzer
        
        # 피드백 루프
        ui_dashboard >> Edge(style="dashed", label="피드백") >> teros_loop
        deliberation >> Edge(style="dashed", label="승인/거부") >> ontology
        context >> Edge(style="dashed", label="맥락 정보") >> analyzer

if __name__ == "__main__":
    print("TEROS 아키텍처 다이어그램 생성 중...")
    create_architecture_diagram()
    print("✅ 다이어그램 생성 완료: TEROS_Architecture.png")

