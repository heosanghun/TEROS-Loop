# TEROS 멀티모달 Agentic AI System 아키텍처

## 전체 시스템 구조도

```mermaid
graph TB
    subgraph "사용자 상호작용 계층 (User Interaction Layer)"
        UI1[교사 대시보드<br/>Teacher Dashboard]
        UI2[학생 대시보드<br/>Student Dashboard]
        UI3[TEROS-Loop 안건 심의<br/>Deliberation Interface]
        UI4[맥락 어노테이션<br/>Context Annotation]
    end

    subgraph "자가 발전 엔진 계층 (Self-Evolving Engine Layer)"
        LOOP[TEROS-Loop<br/>Self-Evolving Engine]
        LOOP1[1단계: 불일치 포착<br/>Discrepancy Detection]
        LOOP2[2단계: 원인 분석 & 지식 추출<br/>LLM Ensemble Analysis]
        LOOP3[3단계: 검증 메커니즘<br/>Reliability-Enhanced Validation]
        BUFFER[개념적 버퍼<br/>Conceptual Buffer]
        ONTOLOGY[교육 지식 온톨로지<br/>Educational Knowledge Ontology]
    end

    subgraph "신뢰성 기반 분석 계층 (Trustworthy Analytics Layer)"
        FAIRNESS[공정성 강화 모듈<br/>Fairness Enhancement Module]
        FAIR1[적대적 학습<br/>Adversarial Learning]
        FAIR2[인과 추론<br/>Causal Inference]
        
        XAI[설명가능성 모듈<br/>Explainability Module]
        XAI1[멀티모달 근거 시각화<br/>Multimodal Saliency Map]
        XAI2[반사실적 설명<br/>Counterfactual Explanations]
        
        CONTEXT[맥락 이해 강화 모듈<br/>Context Awareness Module]
        CONTEXT1[교사 태깅 시스템<br/>Teacher Tagging System]
        CONTEXT2[학생 메타인지 리포트<br/>Student Metacognition Report]
        
        ANALYZER[재능 진단 분석기<br/>Talent Diagnosis Analyzer]
    end

    subgraph "데이터 통합 계층 (Data Integration Layer)"
        COLLECTOR[멀티모달 데이터 수집기<br/>Multimodal Data Collector]
        TEXT[텍스트 데이터<br/>Text Data]
        IMAGE[이미지 데이터<br/>Image Data]
        AUDIO[음성 데이터<br/>Audio Data]
        VIDEO[비디오 데이터<br/>Video Data]
        PREPROCESSOR[데이터 전처리<br/>Data Preprocessing]
    end

    %% 데이터 흐름
    TEXT --> COLLECTOR
    IMAGE --> COLLECTOR
    AUDIO --> COLLECTOR
    VIDEO --> COLLECTOR
    COLLECTOR --> PREPROCESSOR
    PREPROCESSOR --> ANALYZER

    %% 분석 계층 내부 연결
    ANALYZER --> FAIRNESS
    ANALYZER --> XAI
    ANALYZER --> CONTEXT
    FAIRNESS --> FAIR1
    FAIRNESS --> FAIR2
    XAI --> XAI1
    XAI --> XAI2
    CONTEXT --> CONTEXT1
    CONTEXT --> CONTEXT2

    %% 자가 발전 엔진 계층
    ANALYZER --> LOOP
    LOOP --> LOOP1
    LOOP1 --> LOOP2
    LOOP2 --> LOOP3
    LOOP3 --> BUFFER
    BUFFER --> ONTOLOGY
    ONTOLOGY --> ANALYZER

    %% 사용자 상호작용
    ANALYZER --> UI1
    ANALYZER --> UI2
    LOOP3 --> UI3
    UI4 --> CONTEXT
    UI3 --> ONTOLOGY

    %% 피드백 루프
    UI1 -.피드백.-> LOOP
    UI2 -.피드백.-> LOOP
    UI3 -.승인/거부.-> ONTOLOGY

    style LOOP fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    style ONTOLOGY fill:#4ecdc4,stroke:#2d9cdb,stroke-width:2px
    style ANALYZER fill:#95e1d3,stroke:#2d9cdb,stroke-width:2px
    style FAIRNESS fill:#ffe66d,stroke:#f39c12,stroke-width:2px
    style XAI fill:#ffe66d,stroke:#f39c12,stroke-width:2px
    style CONTEXT fill:#ffe66d,stroke:#f39c12,stroke-width:2px
```

## TEROS-Loop 상세 구조

```mermaid
sequenceDiagram
    participant AI as AI 분석 시스템
    participant DETECT as 불일치 포착 엔진
    participant LLM as LLM 앙상블
    participant BUFFER as 개념적 버퍼
    participant TEACHER as 교사
    participant ONTO as 지식 온톨로지

    AI->>DETECT: 예측 결과 전송
    DETECT->>DETECT: 예측-결과 불일치 감지<br/>(KL Divergence 계산)
    
    alt 불일치 발견
        DETECT->>LLM: 불일치 사례 데이터 전송
        LLM->>LLM: 원인 분석 (GPT-4, Llama 3 등)
        LLM->>LLM: 잠정적 지식 규칙 생성<br/>(IF-THEN 형태)
        LLM->>BUFFER: 잠정적 지식 저장
        
        BUFFER->>BUFFER: 베이즈 신뢰도 업데이트<br/>(경험적 누적 검증)
        
        alt 신뢰도 > 임계치(80%)
            BUFFER->>TEACHER: 검증 요청 안건 제시
            TEACHER->>TEACHER: 지식 규칙 검토
            alt 교사 승인
                TEACHER->>ONTO: 지식 통합 승인
                ONTO->>ONTO: OWL 형식으로 변환 및 추가
                ONTO->>AI: 업데이트된 지식 반영
            else 교사 거부/수정
                TEACHER->>BUFFER: 수정 요청 또는 거부
                BUFFER->>BUFFER: 지식 규칙 업데이트/삭제
            end
        end
    end
```

## 멀티모달 데이터 처리 파이프라인

```mermaid
flowchart LR
    subgraph "입력 데이터"
        T1[텍스트<br/>보고서, 메타인지 노트]
        I1[이미지<br/>발표 자료, 작품]
        A1[음성<br/>발표 녹음, 인터뷰]
        V1[비디오<br/>발표 영상, 활동 기록]
    end

    subgraph "데이터 처리"
        T2[텍스트 임베딩<br/>BERT/GPT]
        I2[이미지 특징 추출<br/>CNN/Vision Transformer]
        A2[음성 특징 추출<br/>Whisper/ASR]
        V2[비디오 분석<br/>Temporal CNN]
    end

    subgraph "멀티모달 융합"
        FUSION[멀티모달 융합<br/>Cross-Attention Mechanism]
        CONTEXT_DATA[맥락 정보 통합<br/>Context Annotation]
    end

    subgraph "분석 출력"
        TALENT[재능 프로파일<br/>Talent Profile]
        EVIDENCE[근거 시각화<br/>Evidence Visualization]
    end

    T1 --> T2
    I1 --> I2
    A1 --> A2
    V1 --> V2
    
    T2 --> FUSION
    I2 --> FUSION
    A2 --> FUSION
    V2 --> FUSION
    
    FUSION --> CONTEXT_DATA
    CONTEXT_DATA --> TALENT
    CONTEXT_DATA --> EVIDENCE

    style FUSION fill:#4ecdc4,stroke:#2d9cdb,stroke-width:2px
    style TALENT fill:#95e1d3,stroke:#2d9cdb,stroke-width:2px
```

## 기술 스택 및 구성 요소

### 백엔드
- **AI 모델**: PyTorch, Transformers (Hugging Face)
- **LLM**: GPT-4, Llama 3, 경량화 LLM 앙상블
- **비전 모델**: Vision Transformer, CNN
- **음성 처리**: Whisper, ASR 모델
- **데이터베이스**: PostgreSQL, NoSQL (MongoDB - 개념적 버퍼)

### 프론트엔드
- **프레임워크**: React
- **시각화**: D3.js, Chart.js
- **UI/UX**: 교사/학생 대시보드, 멀티모달 근거 시각화

### 온톨로지
- **형식**: OWL (Web Ontology Language)
- **저장소**: RDF Store
- **추론 엔진**: OWL Reasoner

## 핵심 메커니즘

### 1. 공정성 강화 모듈
- **적대적 학습**: 예측 모델과 적대 모델 동시 학습
- **인과 추론**: 도구 변수 분석을 통한 잠재력 추정

### 2. 설명가능성 모듈
- **멀티모달 그래디언트**: Integrated Gradients, Grad-CAM
- **반사실적 설명**: Genetic Algorithm 기반 탐색

### 3. 맥락 이해 강화 모듈
- **교사 태깅**: 구조화된 맥락 정보 입력
- **학생 메타인지**: BERT 임베딩 기반 통합

### 4. TEROS-Loop
- **1단계**: KL Divergence 기반 불일치 감지
- **2단계**: LLM 앙상블 기반 원인 분석
- **3단계**: 베이즈 신뢰도 업데이트 + 교사 심의

