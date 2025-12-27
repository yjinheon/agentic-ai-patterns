# PreFace

## Agent
AI Agent는 단순히 질문에 답하는 LLM을 넘어서, 환경을 인식하고, 계획을 세우고, 도구를 사용하며, 목표를 달성하기 위해 자율적으로 행동하는 시스템을 의미함

## Agentic Loop

: Agent는 스스로의 행동을 계획하고, 실행하며, 결과를 평가하고, 다음 행동을 결정하는 루프를 거친다.

1. Get the Mission(목표수신)
2. Scan the Scene(환경파악)
3. Think it Through (계획수립)
4. Take Action(실행)
5. Learn and Get Better(학습과 개선)

-> 다시 1로 돌아감

## AI Agent의 4가지 레벨

### Level 0: Core Reasoning Engine 

- LLM 자체
- 현재 학습되어있는 데이터만 활용가능 
- 추론엔진

### Level 1: Connected Problem Solver 

- MCP, Tools
- 외부 도구를 사용가능
- 다단계실행

### Level 2:  Stregetic Problem Solver

- Divide and Conquer. 
- Method Chaing
- Context Engineering 이 추가됨. Prompt를 넘어 AI에게 제공할 정보 환경 자체를 설계 

---

**_Context Engineering_**

- **System Prompt** : AI의 역할과 행동 규칙 정의
- **Retrieved Documents** : Knowledge Base에서 가져온 문서
- **Tool Outputs** : API호출 결과 (캘린더, 날씨 등)
- **Implicit Data** : 사용자 정보, 대화 이력, 환경 상태 

---

## Concepts

---

**_Concept_**

- **AI Agent** : 환경을 인식하고 목표 달성을 위해 자율적으로 계획, 행동, 학습하는 시스템
Agentic Loop: Agent의 핵심 동작 방식. Get Mission → Scan → Think → Act → Learn의 5단계 반복
- **Context Engineering** : AI에게 제공할 정보 환경 전체를 설계하는 방법론. System Prompt + 외부 데이터 + 도구 출력 + 암묵적 정보를 통합
- **Proactive Goal Discovery**: Agent가 사용자의 잠재적 목표를 파악하고 선제적으로 지원하는 능력
- **Metamorphic System**: 목표에 따라 자신의 구조와 구성을 동적으로 변경할 수 있는 시스템

---

# 01. Prompt Chaining Pattern

핵심이 되는 원칙은 아래 두 가지로 요약 할수 있다.

- Divide and Conquer(분할정복)
- Function Composition(함수합성)

## Pattern Overview

- 복잡한 문제를 작은 하위 문제로 분해
- 각 단계는 하나의 명확한 작업에 집중
- 이전 단계의 출력이 다음 단계의 입력이 됨
- 각 단계에서 외부 도구 통합 가능

## Practical Applications & Usecases

1. Information Processing Workflows
2. Complex Query Answering : sequential processing workflow
3. Data Extraction and Transformation
4. Content Generation Workflows
5. Conversational Agents with State
5. Code Generation and Refinement
7. MultiModal and multi-step Reasoning


```md
**Structured Output의 중요성**

- 프롬프트 체인의 신뢰성은 기본적으로 단계간 데이터 전달의 정확성에 달림
- json과 같이 구조화된 형태로 데이터가 전달되어야 신뢰성을 확보할 수 있음
- toon과 같은 context전달을 위한 특수한 자료구조를 검토해볼 수있음
```

## Concepts

---

**_Concept_**

- **Prompt Chaining** : 복잡한 작업을 순차적인 작은 단계로 분해하여 처리하는 패턴. 각 단계의 출력이 다음 단계의 입력이 됨
- **LCEL (LangChain Expression Language)** : LangChain에서 | 파이프 연산자로 컴포넌트를 연결하는 문법
- **Role Assignment** : 각 체인 단계에 "Market Analyst", "Technical Writer" 등 역할을 부여하여 출력 품질 향상
- **Dependency Chain** : 각 단계가 이전 단계의 결과에 의존하는 순차적 처리 흐름

---

# 02. Routing Patterns

Agent에 동적인 의사결정 흐름을 추가하는것

## Routing Pattern Overview

- 적응적 실행: 고정 경로가 아닌 상황에 맞는 경로 선택
- 모듈화: 각 경로를 독립적인 전문 모듈로 구성
- 확장성: 새로운 경로 추가가 용이

**pattern example**


```md
[고객 질문]
      │
      ▼
[의도 분류 (Router)]
      │
      ├─── "주문 상태" ───▶ [주문 DB 조회 Agent/Tool]
      │
      ├─── "제품 정보" ───▶ [제품 카탈로그 검색 Agent/Tool]
      │
      ├─── "기술 지원" ───▶ [트러블슈팅 가이드 Agent/Tool]
      │
      └─── "불명확" ─────▶ [명확화 요청 Agent/Tool]
```

## Routing 전략들

### LLM Based routing
: llm이 자연어를 분석하여 경로 결정. 유연하지만 느림
### Embedding Based routing
: input query를 vector embeddding으로 변환
### Rule Based routing
: 정규식및 패턴 매칭 기반 routing. 빠르고 예측가능하며 구현 쉬움
### Machine Learning Model Based routing
: 별도 학습된 분류모델 활용 경로 결정. 높은 정확도 및 빠른 추론

## Concepts

---

**_Concept_**

- **Routing Pattern**: 입력이나 상태에 따라 워크플로우의 실행 경로를 동적으로 선택하는 패턴
- **Router**: 입력을 분석하고 어떤 경로로 보낼지 결정하는 컴포넌트
- **LLM-based Routing**: LLM이 자연어를 분석하여 경로를 결정하는 방식. 유연하지만 느리고 비용 발생
- **Embedding-based Routing**: 입력과 경로를 벡터로 변환하여 유사도 기반으로 경로 결정. 의미 기반 매칭
- **Rule-based Routing**: 키워드, 패턴, 정규식 등 명시적 규칙으로 경로 결정. 빠르고 예측 가능
- **ML Model-based Routing**: 별도 학습된 분류 모델로 경로 결정. 높은 정확도, 빠른 추론
- **Intent Classification**: 사용자 발화의 의도를 파악하는 것. Routing의 핵심 선행 단계
- **RunnableBranch**: LangChain에서 조건부 분기를 구현하는 컴포넌트
- **Auto-Flow**: Google ADK에서 sub_agents 기반으로 자동 라우팅하는 메커니즘

---

## routing pattern 관련 추가 참조

- **Semantic Router**: 임베딩 기반 라우팅을 쉽게 구현할 수 있는 라이브러리
- **Intent Detection in NLU**: 자연어 이해에서 의도 탐지 기법들
- **Fallback Strategies**: 라우팅 실패 시 처리 방법 (기본 경로, 재시도, 인간 에스컬레이션)
- **Confidence Threshold**: 라우팅 결정의 확신도를 측정하고 임계값 이하면 다른 처리
- **A/B Testing for Routes**: 여러 라우팅 전략의 성능을 비교 테스트하는 방법
- **Hierarchical Routing**: 다단계 라우팅 (대분류 → 중분류 → 소분류)

# 03. Parallelization

독립적인 작업들을 동시에 실행하여 전체 처리 시간을 단축하는 패턴

## Pattern Overview

- Fan-Out: 작업을 여러 병렬 워커에 분배
- Workers: 독립적으로 작업 수행
- Fan-In: 결과를 수집하고 통합

## Concepts

Concept

---

**_Concept_**

- **Parallelization Pattern** : 독립적인 작업들을 동시에 실행하여 처리 시간을 단축하는 패턴
- **Fan-Out**: 입력을 여러 병렬 워커에 분배하는 단계
- **Fan-In**: 병렬 워커들의 결과를 수집하고 통합하는 단계
- **Data Parallelism**: 동일한 작업을 다른 데이터에 병렬 적용(map reduce)
- **Task Parallelism**: 서로 다른 성격의 작업을 동시에 실행 (분석 + 키워드추출 + 요약)
- **RunnableParallel**: LangChain에서 여러 체인을 병렬로 실행하는 컴포넌트
- **asyncio.gather**: Python에서 여러 코루틴을 동시에 실행하고 결과를 수집하는 함수
- **Semaphore**: 동시 실행 수를 제한하는 동기화 도구. Rate Limiting에 활용
- **Sectioning**: 하나의 큰 작업을 여러 관점/섹션으로 나누어 병렬 처리
- **Voting/Ensemble**: 여러 모델/에이전트의 결과를 종합하여 최종 결정

---

## Takeaways
