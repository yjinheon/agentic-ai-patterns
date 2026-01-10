# Prompt Caching

## Overview

- 캐시는 prefix 기반이다. 즉, 프롬프트의 앞부분부터 breakpoint까지가 캐시 단위
- 기본 캐시 수명은 5분 (사용할 때마다 자동 갱신, 추가 비용 없음)
- 캐시는 조직(Organization) 단위로 격리되어 다른 조직과 공유되지 않음
- 캐시 히트를 위해서는 prefix가 100% 동일해야 함


## 왜 Prompt Caching을 사용하는가?


---

**_Concept_**

- **Prompt Caching** : API 요청 시 프롬프트의 특정 prefix(접두사)를 캐시에 저장하여, 이후 동일한 prefix를 포함한 요청에서 재처리 없이 재사용할 수 있게 하는 기능 . 반복되는 긴 컨텍스트를 캐시해두는것
- **cache_control** : 캐시 breakpoint를 지정하는 파라미터 ({"type": "ephemeral"})
- **Cache Hit** : 캐시된 prefix와 일치하여 재사용되는 경우
- **Cache Miss** : 캐시가 없거나 불일치하여 새로 처리 및 저장하는 경우
- **Prefix** : 캐시 단위가 되는 프롬프트의 앞부분 (breakpoint까지)

---


## 02. 기본 캐싱 요청

**첫번째 호출**

```python
import anthropic

client = anthropic.Anthropic()

# 긴 문서 콘텐츠
book_content = "엄청 긴 책의 텍스트 (약 187,000 토큰)"

response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "당신은 텍스트를 분석하는 AI 어시스턴트입니다."
        },
        {
            "type": "text",
            "text": book_content, # 캐싱 대상 긴 문서 콘텐츠
            "cache_control": {"type": "ephemeral"
                              ,"ttl": "1h"}  # 이 지점까지 캐싱, ttl 지정
        }
    ],
    messages=[
        {"role": "user", "content": "이 책의 주요 테마를 분석해주세요."}
    ]
)
```

API응답의 usage 필드에서 캐시 관련 메타데이터 확인 가능

```python
print(response.usage)
```

응답 예시


```bash
{
    "input_tokens": 21, # breakpoint 이후의 토큰 수(캐시 대상 아님)
    "cache_creation_input_tokens": 188086, # 캐시 미스로 새로 캐시에 저장된 
    "cache_read_input_tokens": 0, # 캐시에서 읽은 토큰 수
    "output_tokens": 393
}
```


**두번째 이후  호출**

```bash
{
    "input_tokens": 21,
    "cache_creation_input_tokens": 0, # 캐시 미스로 새로 캐시에 저장된 
    "cache_read_input_tokens": 188086,
    "output_tokens": 393
}

```

## 03. 캐시 구조와 무효화

### Cacheables

Prompt Caching은 요청의 대부분의 블록에 `cache_control`을 지정가능

- Tool 정의 :  `tools` 배열
- System 메시지 :  `system` 배열
-  텍스트 메시지 :  `messages.content`
-  이미지/문서 :  `messages.content`
-  Tool use/result | `messages.content`


### 캐시 계층 구조 (Prefix Hierarchy)

캐시는 다음 순서로 계층적으로 생성됨

```
tools → system → messages
```

**핵심 원칙:** 상위 레벨이 변경되면 하위 레벨 캐시도 모두 무효화

```python
# 요청 구조
{
    "tools": [...],           # 1: tools 캐시
    "system": [...],          # 2: system 캐시 (tools에 의존)
    "messages": [...]         # 3: messages 캐시 (tools + system에 의존)
}
```

tools를 수정하면 system과 messages 캐시 모두 무효화

---

### 캐시 불가능한 요소

일부 요소는 직접 캐싱할 수 없다:

- **Thinking blocks** 
-  **Citations (서브 블록)** 

**Thinking blocks 특수 동작:**
- 직접 캐싱은 불가하지만, 이전 assistant turn에 포함된 thinking block은 다른 콘텐츠와 함께 자동 캐싱됨
- 캐시에서 읽힐 때 input tokens로 계산됨

---

### 캐시 무효화 조건

다음 변경사항은 message 캐시를 무효화한다:

- Tool 정의 수정
- Web search 토글
- Citations 토글
- `tool_choice` 변경
- 이미지 추가/제거
- Thinking 파라미터 변경

**캐시 히트 필수 조건:**

- prefix가 **100% 동일**해야 함
- 동일 위치에 `cache_control` 마커가 있어야 함
- 5분(또는 1시간) 내에 요청해야 함

---

### 실습예제 Tool 정의 캐싱

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=200,
    tools=[
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_database",
            "description": "Search products in database",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            },
            "cache_control": {"type": "ephemeral"}  # 마지막 tool에 캐시 지정
        }
    ],
    messages=[
        {"role": "user", "content": "What's the weather in Seoul?"}
    ]
)
```

Tool 정의가 많고 변경이 적은 경우, 마지막 tool에 `cache_control`을 지정하면 전체 tools 배열이 캐싱됨

---

**_Concept_**

- **캐시 계층 구조** : tools → system → messages 순서로 캐시가 생성되며, 상위 변경 시 하위도 무효화
- **Prefix 기반 캐싱** : 캐시는 처음부터 breakpoint까지의 연속된 prefix를 단위로 저장. 캐시 히트를 위해서는 prefix가 완전히 동일해야 함
- **Thinking block 자동 캐싱** : 직접 캐싱 불가하지만 tool result와 함께 자동으로 캐싱됨
- **캐시 무효화** : tool 정의, 이미지, tool_choice 등의 변경으로 캐시가 무효화됨

---

## 04 캐싱 전략

---

### Breakpoint 배치 전략

 - `cache_control`은 최대 **4개**까지 지정가능.
 - 보통 1개만 써도 충분함

**Breakpoint 1개:**

```python
system=[
    {"type": "text", "text": "시스템 지시사항..."},
    {
        "type": "text", 
        "text": "대용량 문서 전체...",
        "cache_control": {"type": "ephemeral"}  # 마지막에 1개
    }
]
```

**다중 Breakpoint (복잡한 경우):**

다음 상황에서 여러 breakpoint가 유용하다:
- 변경 빈도가 다른 섹션 분리 (예: tools는 거의 안 바뀌고, context는 자주 바뀜)
- 20개 이상의 콘텐츠 블록이 있는 경우
- 편집 가능한 콘텐츠 앞에 breakpoint 배치

```python
system=[
    {
        "type": "text",
        "text": "거의 변경 안 되는 기본 지시사항...",
        "cache_control": {"type": "ephemeral"}  # Breakpoint 1
    },
    {
        "type": "text",
        "text": "자주 업데이트되는 컨텍스트...",
        "cache_control": {"type": "ephemeral"}  # Breakpoint 2
    }
]
```

**핵심:** Breakpoint 자체는 비용이 들지 않는다. 실제로 캐시되거나 읽히는 토큰에만 비용 발생.

---

### 20-block Lookback Window

시스템은 `cache_control` breakpoint로부터 **역방향으로 최대 20개 블록**까지 캐시 히트를 확인한다.

**동작 방식:**

```
블록 1 → 블록 2 → ... → 블록 25 → ... → 블록 30 [cache_control]
                              ↑__________________|
                              20개 블록만 역방향 검사
```

**시나리오별 동작:**

| 시나리오 | 결과 |
|----------|------|
| 블록 30에 breakpoint, 이전 블록 변경 없음 | 블록 30까지 캐시 히트 |
| 블록 25 수정 (breakpoint는 블록 30) | 블록 24까지 캐시 히트, 25-30 재처리 |
| 블록 5 수정 (breakpoint는 블록 30) | **캐시 미스** - 20개 윈도우 밖 |

**20개 초과 블록 해결책:**

편집 가능한 콘텐츠 **앞에** 추가 breakpoint 배치:

```python
messages=[
    # 블록 1-10: 초기 대화
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # ...
    
    # 블록 11 (편집 가능 구간 앞에 breakpoint)
    {
        "role": "user",
        "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
    },
    
    # 블록 12-30: 이후 대화
    # ...
    
    # 블록 30 (마지막 breakpoint)
    {
        "role": "user",
        "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
    }
]
```

---

### 5분 vs 1시간 캐시 선택 기준


| 기준 | 5분 캐시 (기본) | 1시간 캐시 |
|------|----------------|-----------|
| 요청 간격 | < 5분 | 5분 ~ 1시간 |
| 비용 | 기본 × 1.25 (쓰기) | 기본 × 2.0 (쓰기) |
| 갱신 | 사용 시 자동 갱신 (무료) | 사용 시 자동 갱신 (무료) |
| 적합 사례 | 빈번한 대화, 연속 API 호출 | 긴 agent 작업, 간헐적 사용자 응답 |



**5분 캐시 예제:**
```python
"cache_control": {"type": "ephemeral"}
```

**1시간 캐시 사용:**
```python
"cache_control": {"type": "ephemeral", "ttl": "1h"}
```

**TTL 혼합 시 규칙:**
- 긴 TTL(1시간)이 짧은 TTL(5분) **앞에** 와야 함
- 역순 배치 시 오류 발생

```python
system=[
    {
        "type": "text",
        "text": "거의 안 바뀌는 콘텐츠...",
        "cache_control": {"type": "ephemeral", "ttl": "1h"}  # 먼저: 1시간
    },
    {
        "type": "text",
        "text": "자주 바뀌는 콘텐츠...",
        "cache_control": {"type": "ephemeral"}  # 나중: 5분
    }
]
```

---

### 비용 최적화 핵심 원칙

**1. 정적 콘텐츠를 앞에 배치**
```
[자주 안 바뀌는 것] → [가끔 바뀌는 것] → [매번 바뀌는 것]
        ↑                    ↑                  ↑
   cache (1h)           cache (5m)         no cache
```

**2. 캐시 히트율 극대화 체크리스트**
- 시스템 지시사항, 대용량 문서, tool 정의는 프롬프트 앞부분에 배치
- 사용자 메시지(변동성 높음)는 뒤에 배치
- 최소 토큰 수 충족 확인 (Sonnet: 1,024+)

**3. 동시 요청 주의**
- 첫 번째 요청의 **응답이 시작된 후**에야 캐시 엔트리 활성화
- 병렬 요청 시 첫 응답을 기다린 후 나머지 요청 전송

```python
# 잘못된 방식 (캐시 미스 가능)
responses = [client.messages.create(...) for _ in range(5)]  # 동시 전송

# 올바른 방식
first_response = client.messages.create(...)  # 캐시 생성 대기
remaining = [client.messages.create(...) for _ in range(4)]  # 이후 병렬
```

**4. 총 비용 계산**

```
총 비용 = (cache_creation × 1.25) + (cache_read × 0.1) + (input × 1.0)
```

캐시 히트가 많을수록 `cache_read` 비중이 높아져 비용 절감.

---

**_Concept_**

- **Breakpoint** : `cache_control`로 지정하는 캐시 경계점, 최대 4개까지 설정 가능
- **20-block Lookback Window** : breakpoint로부터 역방향 20개 블록까지만 캐시 히트 검사
- **TTL (Time-To-Live)** : 캐시 유효 시간, 5분(기본) 또는 1시간 선택 가능
- **정적 콘텐츠 선배치** : 변경 빈도 낮은 콘텐츠를 앞에 두어 캐시 히트율 극대화
- **캐시 활성화 시점** : 응답 생성이 시작되어야 캐시 엔트리가 활성화됨

---


## 05. 캐싱 유즈케이스별 적용

---

### 대규모 문서 처리

책, 논문, 코드베이스 등 대용량 문서를 분석할 때 가장 효과적인 패턴이다.

**패턴:**

```python
# 문서를 system에 캐싱하고, 다양한 질문을 messages로 전송
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    system=[
        {"type": "text", "text": "You are a document analyst."},
        {
            "type": "text",
            "text": entire_book_content,  # ~100,000+ 토큰
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "3장의 핵심 논점을 요약해줘"}
    ]
)
```

**효과:**
- 첫 요청: 캐시 생성 (전체 문서 처리)
- 이후 요청: 캐시 히트로 **2배 이상 빠른 응답**, **90% 비용 절감**

**적용 사례:**
- 책/논문 Q&A 봇
- 코드베이스 분석 도구
- 팟캐스트/영상 트랜스크립트 분석

---

### Multi-turn 대화

긴 대화에서 이전 컨텍스트를 효율적으로 유지하는 패턴이다.

**패턴: 마지막 user 메시지에 breakpoint**

```python
messages = [
    {"role": "user", "content": "Python에 대해 알려줘"},
    {"role": "assistant", "content": "Python은 고수준 프로그래밍 언어로..."},
    {"role": "user", "content": "역사에 대해 더 알려줘"},
    {"role": "assistant", "content": "1991년 Guido van Rossum이..."},
    # ... 20+ 턴의 대화 ...
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "비동기 프로그래밍은 어떻게 해?",
                "cache_control": {"type": "ephemeral"}  # 대화 끝에 breakpoint
            }
        ]
    }
]
```

**점진적 캐싱 전략:**

대화가 길어질수록 중간에 breakpoint를 추가:

```python
messages = [
    # 턴 1-10
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # ...
    
    # 턴 10에 중간 breakpoint
    {
        "role": "user",
        "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
    },
    
    # 턴 11-20
    # ...
    
    # 마지막 턴에 최종 breakpoint
    {
        "role": "user",
        "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
    }
]
```

---

### Tool 정의 캐싱

많은 tool을 정의한 경우, tool 배열 전체를 캐싱한다.

**패턴:**

```python
tools = [
    {"name": "tool_1", "description": "...", "input_schema": {...}},
    {"name": "tool_2", "description": "...", "input_schema": {...}},
    # ... 많은 tool 정의 ...
    {
        "name": "tool_n",
        "description": "마지막 tool",
        "input_schema": {...},
        "cache_control": {"type": "ephemeral"}  # 마지막 tool에 지정
    }
]

response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=500,
    tools=tools,
    messages=[{"role": "user", "content": "서울 날씨 알려줘"}]
)
```

**효과:**
- Tool 정의가 변경되지 않는 한 모든 요청에서 캐시 히트
- Agentic 시스템에서 반복적인 tool 정의 처리 비용 절감

---

### Agentic Tool Use

Agent가 여러 번의 tool call을 수행하는 워크플로우에서 캐싱 적용.

**일반적인 Agent 루프:**

```
User → Agent → Tool Call 1 → Tool Result 1 
            → Tool Call 2 → Tool Result 2
            → ... 
            → Final Response
```

각 단계마다 API 호출이 발생하므로, 누적 컨텍스트 캐싱이 중요하다.

**패턴:**

```python
def agent_loop(user_query):
    messages = [{"role": "user", "content": user_query}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1000,
            tools=tools,  # 캐시된 tool 정의
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=messages
        )
        
        # tool_use가 있으면 실행 후 결과 추가
        if has_tool_use(response):
            messages.append({"role": "assistant", "content": response.content})
            tool_result = execute_tool(response)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result["id"],
                        "content": tool_result["output"],
                        "cache_control": {"type": "ephemeral"}  # tool result에 캐싱
                    }
                ]
            })
        else:
            return response  # 최종 응답
```

**1시간 캐시 권장 사례:**
- Agent 작업이 5분 이상 소요될 수 있는 경우
- 사용자 응답 대기 시간이 불확실한 경우

---

### Extended Thinking 연동

Extended thinking 사용 시 thinking block의 특수 캐싱 동작을 이해해야 한다.

**핵심 규칙:**
1. Thinking block은 직접 `cache_control` 지정 불가
2. Tool result만 전달 시 thinking block이 함께 자동 캐싱됨
3. 일반 user 메시지 추가 시 이전 thinking block 모두 제거됨

**예시 흐름:**

```python
# 요청 1: 사용자 질문
# 응답 1: [thinking_block_1] + [tool_use_1]

# 요청 2: tool_result 전달 (캐시 유지)
messages = [
    {"role": "user", "content": "파리 날씨 알려줘"},
    {"role": "assistant", "content": [thinking_block_1, tool_use_1]},
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "...",
                "content": "15°C, 맑음",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
]
# → thinking_block_1 포함하여 캐싱됨

# 요청 3: 일반 텍스트 추가 (캐시 무효화)
messages.append({
    "role": "user",
    "content": "고마워!"  # 일반 텍스트 → 모든 thinking block 제거
})
# → 이전 thinking block이 컨텍스트에서 제거됨
```

**주의:** 일반 user 메시지가 추가되면 새로운 assistant loop가 시작되고, 이전 thinking block은 모두 무시된다.

---

**_Concept_**

- **대규모문서캐싱** : 문서를 system에 배치하고 다양한 질문을 messages로 전송하는 패턴
- **Multi-turn 캐싱** : 대화 끝에 breakpoint를 두어 누적 컨텍스트 재사용
- **Tool정의캐싱** : 마지막 tool에 cache_control 지정하여 전체 tools 배열 캐싱
- **Agent루프캐싱** : tool result에 cache_control 지정하여 중간 상태 캐싱
- **Thinking block 자동 캐싱** : tool result 전달 시 함께 캐싱, 일반 메시지 추가 시 제거

---

## 06. 트러블슈팅하기 


**캐시 미생성 원인**
- 최소 토큰 미달 (Sonnet: 1,024+)
- 지원 안 되는 모델

**캐시 히트 실패 원인**
- prefix 100% 불일치
- TTL 만료 (5분/1시간)
- 병렬 요청 시 캐시 미활성화
- tool_choice, 이미지 변경
- JSON 키 순서 불안정

### Prompt Caching Utility

**캐시 히트율 계산**


```python
def calculate_cache_hit_rate(usage):
    cached = usage.cache_read_input_tokens
    created = usage.cache_creation_input_tokens
    total_cacheable = cached + created
    
    if total_cacheable == 0:
        return 0
    
    return (cached / total_cacheable) * 100

```

**캐시 히트율 계산**


```python
def calculate_cache_hit_rate(usage):
    cached = usage.cache_read_input_tokens
    created = usage.cache_creation_input_tokens
    total_cacheable = cached + created
    
    if total_cacheable == 0:
        return 0
    
    return (cached / total_cacheable) * 100

```




**실무 체크리스트**
- 정적 콘텐츠 앞에 배치
- 최소 토큰 충족 확인
- tool_choice, 이미지 일관성 유지
- 첫 요청에서 cache_creation, 두 번째에서 cache_read 확인
- 캐시 히트율 정기 모니터링
