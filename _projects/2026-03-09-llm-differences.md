---
title: "PoC NL2SQL Framework - LLM 증축 내용"
date: 2026-03-09 09:00:00 +0900
summary: "NL2SQL PoC LLM augmentation schedule"
categories: [llm_nl2sql, nl2sql]
layout: page
---

<style>
pre {
  overflow-y: auto;
  max-height: 320px;
}
</style>

# PoC NL2SQL Framework - LLM 증축 내용

> 멀티턴 에이전트 기준 적용 방식 설명
> “휴리스틱 중심 파이프라인을 기준으로 LLM을 붙이는 구조”로 설계

## 1. 적용 범위를 코드 관점으로만 요약

멀티턴 에이전트는 다음 구간으로 구성되어 있다.
- `prepare`
- `classify_intent`
- `nl2sql_router`
- `run_nl2sql`
- `show_sql` / `describe_schema` / `explain_last`

LLM은 이 중 `prepare`, `classify_intent`, `nl2sql_router`, SQL 생성 보조 단계에만 걸리며,
파이프라인의 실행/캐시/감사/메트릭은 그대로 둔다.

---

## 2. LLM 적용 위치별 상세

### 2-1. Query Preparation 단계

입력 문장을 `current_query`로 정제하고, 보조 질문(`pending`)을 이어 붙이는 구간이다.

적용 포인트:
- 누락된 slot(예: 지표/대상 단위) 추론 보조
- 표현 정규화(동의어/약어/구문 변형)
- 이전 `messages`를 참조해 follow-up 병합 강화

### 2-2. Intent 분류 단계

분류 대상은 기존처럼 대분류로 유지한다.
- `show_sql`
- `describe_schema`
- `explain_last`
- `nl2sql`

LLM 분류는 아래를 추가한다.
- 대화 히스토리 요약 컨텍스트 반영
- `show_sql`/`describe_schema`와 같은 관리형 요청을 과잉 해석하지 않도록 guardrail
- 불명확한 경우도 `clarification` 후보로 흘려 예측 위험을 낮춤

### 2-3. NL2SQL Router 단계

이 단계에서 “즉시 실행”과 “질문 되묻기”를 구분한다.

- 기존: 휴리스틱 규칙
- 증축: LLM이 reasoner 역할, 다만 policy/JSON schema는 엄격하게 제한

출력은 구조화된 형태(`intent`, `need_metric`, `pending_reason`, `next_action`)를 유지해
`run_nl2sql` 혹은 `ask clarification`로 분기한다.

### 2-4. SQL 생성 보조

템플릿이 처리하기 어려운 표현에 대해서만 LLM 제안을 받아 템플릿 입력값이나 필드명을 조정한다.

주의할 점:
- 실행되는 최종 SQL은 기존 실행 정책(권한/검증/파라미터 타입/제한) 아래에서만 반영
- LLM 출력은 `raw_sql` 그대로 실행하지 않고 `schema aware` 정규화 단계에서 한 번 더 정제

---

## 3. 운영 안정성을 위한 레이어

### 3-1. Feature Flag & Fallback

`LLM_ENABLE_*` 플래그로 각 단계 독립 토글.
- 실패 시 heuristic 경로로 자동 복귀
- 디버그 모드에서 decision trace를 남겨 비교 분석

### 3-2. Thread 상태와 멀티턴 결합

`thread_id` 기반 상태관리 덕분에 다음이 가능해진다.
- 보류 질문(`pending`)의 안전한 보관
- follow-up에서 이전 쿼리 맥락 재사용
- `thread reset` 및 상태 조회(admin용)로 운영 디버깅

### 3-3. 관측성(Observability)

- request_id 기반 추적
- `latency`/`nl2sql_requests`/`cache_hit` 지표
- 각 단계 intent·pending 이력을 메트릭/로그에 남김

---

## 4. 왜 이 방식이 유지보수에 유리한가

- 기능 스위칭이 빠르다: LLM 문제 발생 시 해당 단계만 끄고 heuristic만 유지 가능
- 단계별 성능 비용이 보이는 구조다: 어떤 단계가 병목인지 분리 측정 가능
- 평가를 통해 점진적으로 신뢰구간 확대

---

## 5. 실제 작동 예시

### 단발성 Query
- 처리 전 query: "챗지피티야 다운타임 많은 장비 10개 보여줘 알겠지?"
- 처리 후 query: "다운타임 많은 장비 top10"

```bash
{
  "ok": true,
  "request_id": "b7551fa9-7b3e-4439-ba1a-ad961f90bd97",
  "kg_version": "8a764a11-d30c-42b3-a6f9-36582e4a1e70",
  "query_original": "챗지피티야 다운타임 많은 장비 10개 보여줘 알겠지?",
  "query_prepared": "다운타임 많은 장비 top10",
  ..., # 생략
  "sql": "SELECT a.asset_name,\n               ROUND(SUM((julianday(e.end_ts) - julianday(e.start_ts)) * 24.0), 3) AS downtime_hours\n        FROM events e\n        JOIN assets a ON a.asset_id = e.asset_id\n        WHERE e.event_type = 'downtime' AND e.end_ts IS NOT NULL\n        GROUP BY a.asset_name\n        ORDER BY downtime_hours DESC\n        LIMIT 10;",
  ..., # 생략
  "rows": [
    {
      "asset_name": "CMP-074",
      "downtime_hours": 32.75
    },
    {
      "asset_name": "DIFF-183",
      "downtime_hours": 32.25
    },
    {
      "asset_name": "CMP-002",
      "downtime_hours": 31.117
    },
    {
      "asset_name": "DIFF-123",
      "downtime_hours": 30.65
    },
    {
      "asset_name": "CMP-169",
      "downtime_hours": 29.75
    },
    {
      "asset_name": "ETCH-129",
      "downtime_hours": 29.233
    },
    {
      "asset_name": "CVD-086",
      "downtime_hours": 29.233
    },
    {
      "asset_name": "CVD-189",
      "downtime_hours": 29.2
    },
    {
      "asset_name": "DIFF-118",
      "downtime_hours": 28.9
    },
    {
      "asset_name": "DIFF-038",
      "downtime_hours": 28.583
    }
  ],
  "timings_ms": {
    "rewrite": 17.389338463544846,
    "kg_context": 3.909321501851082,
    "sqlgen": 696.0063045844436,
    "execute": 13.33877258002758,
    "total": 730.6495551019907
  },
  ...
```

### Multi turn query
#### 첫 쿼리: 불완전 쿼리
- 입력 query: "장비 top10"
  - 어떤 기준을 적용해 top 10을 뽑는지 알 수 없으므로 불완전한 query
- 처리 결과: 추가 질문 제안
  - "TOP/상위 질의는 기준(지표)이 필요합니다. 예: `다운타임 많은 장비 top10`, `온도 높은 장비 top10`. 어떤 지표로 볼까요?"
  - pending에 추가 질문이 필요한 근거, 기존 query를 저장함

```bash
{
  "ok": true,
  "request_id": "8d775ea9-971e-4927-b896-56089bf27f3c",
  "thread_id": "469d7d67-e897-4d24-a294-dbe8e96567f9",
  "assistant_message": "TOP/상위 질의는 기준(지표)이 필요합니다. 예: `다운타임 많은 장비 top10`, `온도 높은 장비 top10`. 어떤 지표로 볼까요?",
  "intent": "nl2sql",
  "current_query": "장비 top10",
  "pending": {
    "reason": "need_metric",
    "base_query": "장비 top10"
  },
  "last_nl2sql_ok": null,
  "last_nl2sql": null,
  "messages": [
    {
      "role": "human",
      "content": "장비 top10"
    },
    {
      "role": "ai",
      "content": "TOP/상위 질의는 기준(지표)이 필요합니다. 예: `다운타임 많은 장비 top10`, `온도 높은 장비 top10`. 어떤 지표로 볼까요?"
    }
  ]
}
```
#### 다음 쿼리: 정보 보충
- 입력 query: "다운타임"
- 처리 결과: 추가 정보 조회 결과
  - `thread_id`가 유지되어, 지난 대화에서 정보를 보충할 수 있다.

```bash
{
  "ok": true,
  "request_id": "4d701094-6b7a-4a8a-abb2-8782aae23a32",
  "thread_id": "469d7d67-e897-4d24-a294-dbe8e96567f9",
  ..., #생략
    "rows": [
      {
        "asset_name": "CMP-074",
        "downtime_hours": 32.75
      },
      ...,
      {
        "asset_name": "DIFF-038",
        "downtime_hours": 28.583
      }
    ],
  ...,#생략
  },
  "messages": [
    {
      "role": "human",
      "content": "장비 top10"
    },
    {
      "role": "ai",
      "content": "TOP/상위 질의는 기준(지표)이 필요합니다. 예: `다운타임 많은 장비 top10`, `온도 높은 장비 top10`. 어떤 지표로 볼까요?"
    },
    {
      "role": "human",
      "content": "다운타임"
    },
    {
      "role": "ai",
      "content": "다운타임 TOP 결과 (hours):\n1. CMP-074: 32.75h\n2. DIFF-183: 32.25h\n3. CMP-002: 31.117h\n4. DIFF-123: 30.65h\n5. CMP-169: 29.75h\n6. ETCH-129: 29.233h\n7. CVD-086: 29.233h\n8. CVD-189: 29.2h\n9. DIFF-118: 28.9h\n10. DIFF-038: 28.583h\n\n후속 예: `1등 장비 이벤트 보여줘`, `ETCH-01 다운타임 보여줘`"
    }
  ]
}
```