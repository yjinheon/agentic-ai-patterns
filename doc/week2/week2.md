# 04. Reflection Pattern

- iteratively self-correct and refine outputs
- involves feedback loop of execution, evaluation, refinement
- tradeoff : incresed latency, computational expense

## Pattern Overview

- llm이 스스로 출력을 검토하고 개선하는것(Self Correction)
- 출력물에 대한 일종의 퇴고 절차를 추가하는 것
- Generate -> Critique -> Refine 사이클
- 품질이 일정 기준을 넘어설 때까지 사이클을 반복

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  [입력] ──▶ [초기 생성] ──▶ [자기 평가] ──▶ [개선] ──▶ [출력]    │
│                                │                                │
│                                │ 문제 발견?                      │
│                                │                                │
│                                └──── Yes ────┐                  │
│                                              │                  │
│                                              ▼                  │
│                                        [재생성]                  │
│                                              │                  │
│                                              └──▶ [자기 평가]    │
│                                                                 │
│                          반복 (Iteration)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### reflection pattern의 3 가지 컴포넌트

1. Generator : 초기 출력생성
2. Reflector : 출력평가 및 피드백
3. Refiner : 피드백 반영 및 개선

## 3가지 Reflection 구현 방식

### 1.Single Agent Reflection

: 하나의 LLM이 생성과 평가를 모두 진행

### 2.Multi Agent Reflection
:  서로 다른 Agent가 각각의 역할을 담당함

ex) 초안을 작성하는 agent, 문서를 피드백하는 agent, 피드백을 반영하는 agent가 따로 존재함

**도구나 외부 시스템**의 피드백 활용 개선

```md
[생성된 코드]
    │
    ├──▶ [Unit Test 실행] ──▶ "2/5 테스트 실패"
    │
    ├──▶ [Linter 실행] ──▶ "unused variable on line 3"
    │
    └──▶ [Type Checker] ──▶ "type mismatch on line 7"
    
         │
         ▼
    [구체적인 오류 정보로 수정]
```

### 3.Rubric-Based Reflection(기준 기반 평가)

: 명시적 평가 기준을 제공하는것

애매한 부분이 어느정도 존재하는건 감안해야함.

```
rubric = """
다음 기준으로 1-5점 평가하세요:

1. 정확성 (Accuracy): 사실적 오류 없음
2. 완성도 (Completeness): 모든 요구사항 충족
3. 코드 품질 (Quality): 가독성, 효율성
4. 문서화 (Documentation): 주석과 설명 적절함

각 항목이 4점 이상이면 통과, 아니면 개선 필요.
"""
```

## Reflection 구현 시 고려사항

### 1. 무한루프 방지

```python
MAX_ITERATIONS = 5

def should_continue(state):
    if state["iteration"] >= MAX_ITERATIONS:
        return "end" 
    # ...
```


### 2. 수렴하지 않을 경우의 분기 
연속으로 기준 점수가 개선되지 않을 경우 종료하는 로직

```pyhon
def should_continue(state):
    # 이전 점수와 비교
    if state["score"] <= state.get("prev_score", 0):
        state["no_improvement_count"] = state.get("no_improvement_count", 0) + 1
        if state["no_improvement_count"] >= 2:
            return "end"  # 개선 없음, 종료
```

### 3. 호출비용

llm이 생성한 출력을 llm이 평가함-> 인간의 검토 프로세스를 어느 정도 llm에게 전가한 것. 
기본적으로 같은 task에 대해 검토관련 호출이 추가되기 때문에 호출 비용이 추가적으로 들어간다.

## Concepts

---

**_Concept_**

**Reflection Pattern**: LLM이 자신의 출력을 평가하고 개선하는 자기 개선 패턴. 실행-> 평가-> 개선의 피드백 루프를 거침
**Self-Correction**: Reflection을 통해 오류를 스스로 발견하고 수정하는 능력
**Iterative Refinement**: 만족스러운 품질에 도달할 때까지 생성-평가-개선을 반복

---



# 05. Tool Use Pattern

## Pattern Overview

- llm이 외부 API, DB, 서비스와 상호작용하여 업무의 범위를 확장하는 패턴

## Tool

- Tool은 식별자, 설명, 파라미터로 이루어진 llm이 외부 api와 상호작용하기 위한 일종의 명세


```python

## Tool 정의 예시
tool_definition = {
    # 1. 이름 (Name)
    "name": "search_database",
    
    # 2. 설명 (Description) 
    "description": """
    고객 데이터베이스에서 정보를 검색합니다.
    고객 이름, 이메일, 주문 내역을 조회할 때 사용
    실시간 고객 정보가 필요할 때 이 도구를 호출
    """,
    
    # 3. 파라미터
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색할 고객 이름 또는 이메일"
            },
            "limit": {
                "type": "integer",
                "description": "반환할 최대 결과 수",
                "default": 10
            }
        },
        "required": ["query"]
    }
}
```

**Description의 중요성**

- LLM은 description을 읽고 **언제** 이 도구를 사용할지 결정
- 명확하고 구체적인 설명이 정확한 도구 선택으로 이어짐


### Tool 유형별 분류

**1. 정보 검색 도구 (Retrieval Tools)**

- 웹 검색
- DB 쿼리
- 파일읽기
- Api데이터 가져오기
- 벡터DB검색


**2. 행동 도구 (Action Tools)**

- 이메일 발송
- 일정 생성/수정
- 파일 생성/저장
- 외부서비스 호출
- Code Executor

**3. 연산 도구 (Computation Tools)**


-  Calculator
- Unit Converter
- Data Analyzer
- Code Interpreter

---

### Tool Use 고려사항

**1. 명확한 description 작성하기**


```md
# simple 
"description": "데이터 검색"

# Verbose
"description": "이 도구는 데이터베이스에서 정보를 검색하는 기능을 
제공하며, SQL을 사용하여 다양한 테이블에서 데이터를 가져올 수 
있습니다. 사용자가 고객 정보, 주문 내역, 제품 정보 등을..."

# 적절함
"description": """
고객 데이터베이스를 검색합니다.
사용 시점: 고객 이름, 이메일, 주문 내역 조회 시
입력: 고객 이름 또는 이메일
반환: 고객 정보 및 최근 주문 5건
"""
```


**2. 권한관리**

```python
# 위험한 도구는 확인 절차 추가
@tool
def delete_file(filepath: str) -> str:
    """파일을 삭제합니다. (주의: 복구 불가)"""
    
    # Human-in-the-Loop: 사용자 확인 요청
    confirmation = request_user_confirmation(
        f"정말로 {filepath}를 삭제하시겠습니까?"
    )
    
    if confirmation:
        os.remove(filepath)
        return f"{filepath} 삭제 완료"
    return "삭제가 취소됨."

```


## Concept

---

## _Concept_

- **Tool Use Pattern**: LLM이 외부 도구를 호출하여 자신의 능력을 확장하는 패턴
- **Function Calling**: LLM이 구조화된 형식으로 함수 호출을 생성하는 기능.
- **Tool Definition**: 도구의 이름, 설명, 파라미터 스키마로 구성된 도구 명세
- **Tool Description**: LLM이 도구를 언제 사용할지 판단하는 핵심 정보. 명확하고 구체적이어야 함
- **Tool Selection**: 여러 도구 중 적절한 도구를 선택하는 과정. Description 품질에 크게 의존함
- **Multi-Tool Orchestration**: 복잡한 작업을 위해 여러 도구를 순차적/병렬로 조합하여 사용
- **ToolMessage**: 도구 실행 결과를 LLM에게 전달하는 메시지 형식
- **bind_tools()**: LangChain에서 LLM에 도구를 연결하는 메서드
- **FunctionTool**: Google ADK에서 Python 함수를 도구로 변환하는 클래스
- **Sandbox Executor**: 코드와 실행도구의 보안을 위한 격리환경

---


# 06. Planning Pattern

## Pattern Overview

- 기본적으로 복잡한 목표는 여러 단계의 계획이 필요함
- Planning Pattern은 Agent가 목표를 분석하고 단계별 계획을 수립하며 체계적으로 실행할 수 있게 함


## Planning의 두 가지 주요 접근법


### 1. Plan and Execute
: 전체 계획을 미리 수립하고 순차적으로 접근하는 패턴

### 2. ReAct(Reasoning + Acting)

: 추론과 행동을 번갈아가며 수행하며 동적으로 적응함


---

**_Concept_**

- **Planning Pattern**: 목표 달성을 위해 단계별 계획을 수립하고 실행하는 패턴
- **Plan-then-Execute**: 전체 계획을 먼저 수립한 후 순차적으로 실행하는 접근법
- **ReAct (Reasoning + Acting)**: 추론(Thought)과 행동(Action)을 번갈아 수행하며 동적으로 적응하는 접근법

---

# 07. Multi Agent Collaboration Pattern

Agent간 전문화, 협업, 분업

## Pattern Overview

전문화된 여러 Agent가 역할을 분담하고 협업하여 복잡한 목표를 달성하는 패턴


## Multi Agent Architecture유형

### 1. 계층형

최상위 Agent가 하위 Agent들을 관리

```text

                    ┌─────────────┐
                    │  Manager    │
                    │   Agent     │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌───────────┐  ┌───────────┐  ┌───────────┐
      │  Worker   │  │  Worker   │  │  Worker   │
      │  Agent 1  │  │  Agent 2  │  │  Agent 3  │
      └───────────┘  └───────────┘  └───────────┘


```

### 2. Flat

```text

      ┌───────────┐       ┌───────────┐
      │  Agent A  │◄─────►│  Agent B  │
      └─────┬─────┘       └─────┬─────┘
            │                   │
            │   ┌───────────┐   │
            └──►│  Agent C  │◄──┘
                └───────────┘

```



### 3. Sequential

Agent들이 파이프라인처럼 순차적으로 작업

```text

┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Agent 1 │───►│ Agent 2 │───►│ Agent 3 │───►│ Agent 4 │
│ Research│    │  Draft  │    │  Edit   │    │ Publish │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

```

- prompt chain과 유사하지만 각 단계가 독립적인 Agent임



### 4. Hybrid 

여러 패턴을 조합 하는 아키텍처



```text

                    ┌─────────────┐
                    │  Manager    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌───────────┐  ┌───────────┐  ┌───────────┐
      │ Research  │  │ Research  │  │  Review   │
      │ Agent 1   │  │ Agent 2   │  │  Agent    │
      └─────┬─────┘  └─────┬─────┘  └───────────┘
            │              │              ▲
            └──────┬───────┘              │
                   ▼                      │
             ┌───────────┐                │
             │  Writer   │────────────────┘
             │  Agent    │
             └───────────┘

```

### **역할 설계 원칙:**

명확한 책임: 각 Agent의 역할이 겹치지 않게
적절한 권한: 필요한 도구만 부여
맥락 있는 배경: backstory로 일관된 행동 유지


## Multi-Agent 통신 패턴

1. 메시지 전달

```json
# Agent 간 메시지 교환
message = {
    "from": "Researcher",
    "to": "Writer",
    "type": "research_complete",
    "content": {
        "topic": "AI 트렌드",
        "findings": [...],
        "sources": [...]
    }
}

```


2. 상태 공유

```json
shared_state = {
    "goal": "블로그 포스트 작성",
    "research_data": None,      # Researcher가 채움
    "draft": None,              # Writer가 채움
    "review_feedback": None,    # Reviewer가 채움
    "final_output": None        # 최종 결과
}

```

## Concept

---

**_Concept_**


- **Multi-Agent Collaboration** : 전문화된 여러 Agent가 역할을 분담하고 협업하여 복잡한 목표를 달성하는 패턴
- **Orchestrator/Manager Agent**: 다른 Agent들의 작업을 조율하고 관리하는 상위 Agent
- **Delegation**: 한 Agent가 다른 Agent에게 작업을 위임하는 것
- **CrewAI**: Multi-Agent 협업에 특화된 프레임워크
- **Role, Goal, Backstory**: CrewAI에서 Agent의 역할, 목표, 배경 스토리를 정의하는 요소
- Agent Communication Languages (ACL): Agent 간 표준 통신 프로토콜

---


# 08. Memory Management 

## Pattern Overview

- LLM은 기본적으로 Stateless한 추론 엔진임
- 지속가능한 작업을 하려면 context를 메모리에 저장할 수단이 필요함
- Memory Management는 Agent가 대화 컨텍스트와 정보를 어떻게 유지하게 할 것이냐의 문제

---


## Memory의 유형

### Short-Term Memory

- 현재 대화 세션의 컨텍스트
- 최근 N개의 메시지
- 세션 종료시 소멸


### long-Term Memory

- 세션 간 지속되는 정보
- 사용자 선호도, 과거 상호작용
- 외부 저장소에 영구보관


### Working Memory

- 현재 작업에 필요한 임시 정보
- 중간 계산 결과, 작업 상태 
-  작업 완료 시 폐기 가능


## Short-term Memory 구현

### 1. Full History
: 모든 대화를 그대로 유지

- Context Window 한계에 빠르게 도달함


### 2. Sliding Window (슬라이딩 윈도우)

: 최근 N개 메시지만 유지

```python
MAX_MESSAGES = 10

def add_message(messages, new_message):
    messages.append(new_message)
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]  # 최근 10개만 유지
    return messages
```

- 단순, 메모리 효율적
- 오래된 중요 정보 손실가능성

---

### 3. Token-based Truncation

: 토큰 수 제한으로 관리

```python
MAX_TOKENS = 4000

def truncate_by_tokens(messages, max_tokens):
    total_tokens = 0
    result = []
    
    # 최신 메시지부터 역순으로
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        result.insert(0, msg)
        total_tokens += msg_tokens
    
    return result
```

---

### 4. Summary Memory

: 오래된 대화를 요약으로 압축



```python
def summarize_and_truncate(messages, summary_threshold=20):
    if len(messages) <= summary_threshold:
        return messages
    
    # 오래된 메시지들 요약
    old_messages = messages[:-10]
    recent_messages = messages[-10:]
    
    summary = llm.invoke(f"다음 대화를 핵심만 요약하세요:\n{old_messages}")
    
    return [
        {"role": "system", "content": f"이전 대화 요약: {summary}"},
        *recent_messages
    ]
```


## Long-term Memory 구현

: 세션 간 지속되는 정보 저장


### Memory 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    [사용자 입력]                                                 │
│         │                                                       │
│         ▼                                                       │
│    ┌─────────────────────────────────────────┐                  │
│    │           Memory Manager                │                  │
│    │                                         │                  │
│    │  1. 관련 기억 검색 (Retrieval)           │                  │
│    │  2. 컨텍스트에 추가                      │                  │
│    │  3. LLM 호출                            │                  │
│    │  4. 새 정보 저장                         │                  │
│    └─────────────────────────────────────────┘                  │
│         │                   │                                   │
│         ▼                   ▼                                   │
│    ┌─────────┐        ┌─────────────┐                          │
│    │Short-term│       │  Long-term  │                          │
│    │ Memory  │        │   Memory    │                          │
│    │         │        │             │                          │
│    │ 현재    │        │ • DB        │                          │
│    │ 대화    │        │ • Vector    │                          │
│    │         │        │   Store     │                          │
│    └─────────┘        └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


## Memory 설계 고려사항

**1. 무엇을 기억할 것인가?**
- 모든 대화 vs 핵심 정보만
- 사실 정보 vs 감정/선호도

**2. 얼마나 오래 기억할 것인가?**
- 세션 내 vs 영구 저장
- 만료 정책 (TTL)

**3. 어떻게 검색할 것인가?**
- 최신순 vs 관련도순
- 키워드 vs 의미 기반

**4. 프라이버시**
- 민감 정보 처리
- 데이터 삭제 정책

---

## _Concept_

- **Memory Management**: Agent가 대화 컨텍스트와 정보를 유지하고 활용하는 패턴
- **Short-term Memory**: 현재 대화 세션 내의 컨텍스트. 세션 종료 시 소멸
- **Long-term Memory**: 세션 간 지속되는 정보. 외부 저장소에 영구 보관
- **Working Memory**: 현재 작업에 필요한 임시 정보
- **Sliding Window**: 최근 N개 메시지만 유지하는 전략
- **Summary Memory**: 오래된 대화를 요약하여 압축하는 전략
- **Vector Store Memory**: 임베딩 기반으로 관련 기억을 의미 검색하는 방식
- **Entity Memory**: 대화에서 언급된 엔티티(사람, 장소 등)를 추적하는 방식
- **Checkpointer**: LangGraph에서 상태를 영속화하는 컴포넌트
- **Semantic Memory**: 일반적 지식과 개념을 저장하는 방식

---

# 09. Learning and Adaptation

## Pattern Overview

- 기본 Agent는 정적이기 때문에 이전 경험에서 학습하고 행동을 개선하게끔 조정해줄 필요가 있음
- 강화학습의 일종

## Agent Learning 방식들

### 1. In-Context Learning

- Few-shot 예시 제공                                       
- 동적 프롬프트 조정                                       
- 모델 가중치 변경 없음                                    
                                                          
###  2. Feedback-based Adaptation

- 사용자 피드백 수집                                 
- 선호도 학습                                        
- 행동 패턴 조정                                     
                                                          
###  3. Experience Replay

- 과거 성공/실패 사례 저장                           
- 유사 상황에서 참조                                 

```python
pythonfrom langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class ExperienceMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.experience_store = None
        self.experiences = []
    
    def store_experience(self, task, action, outcome, success: bool):
        experience = {
            "task": task,
            "action": action,
            "outcome": outcome,
            "success": success
        }
        self.experiences.append(experience)
        
        # 벡터 스토어 업데이트
        text = f"Task: {task}\nAction: {action}\nSuccess: {success}"
        if self.experience_store is None:
            self.experience_store = FAISS.from_texts([text], self.embeddings)
        else:
            self.experience_store.add_texts([text])
    
    def retrieve_relevant_experiences(self, current_task, k=3):
        if self.experience_store is None:
            return []
        
        # 유사한 과거 경험 검색
        results = self.experience_store.similarity_search(current_task, k=k)
        return results
    
    def get_guidance(self, current_task):
        experiences = self.retrieve_relevant_experiences(current_task)
        
        successes = [e for e in experiences if "Success: True" in e.page_content]
        failures = [e for e in experiences if "Success: False" in e.page_content]
        
        guidance = ""
        if successes:
            guidance += f"과거 성공 사례 참고:\n{successes[0].page_content}\n"
        if failures:
            guidance += f"과거 실패 사례 (피하세요):\n{failures[0].page_content}\n"
        
        return guidance



```


                                                          
###  4. Fine-tuning

- 모델 가중치 업데이트                               
- 도메인 특화                                        
- 오프라인 학습                                      
                                                     
---

## _Concept_

- **Learning and Adaptation**: Agent가 경험에서 학습하고 행동을 개선하는 패턴
- **RLHF (Reinforcement Learning from Human Feedback)**: 인간 피드백으로 강화학습
- **In-Context Learning**: 프롬프트에 예시를 제공하여 모델 행동을 조정. 가중치 변경 없음
- **Feedback-based Adaptation**: 사용자 피드백을 수집하고 향후 응답에 반영
- **Experience Replay**: 과거 성공/실패 경험을 저장하고 유사 상황에서 참조
- **Self-Improvement Loop**: 실행 → 평가 → 학습의 반복 사이클
- **Prompt Optimization**: 테스트 케이스 기반으로 프롬프트를 자동 개선
- **Fine-tuning**: 모델 가중치를 업데이트하여 도메인에 특화

---
