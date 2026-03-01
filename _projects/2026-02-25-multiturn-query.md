---
title: "PoC NL2SQL Framework - Multiturn Query 구현"
date: 2026-02-25 09:00:00 +0900
summary: "NL2SQL PoC Multiturn Query Demo"
categories: [heuristic_nl2sql, nl2sql]
---

<style>
pre {
  overflow-y: auto;
  max-height: 320px;
}
</style>

## Multi-Turn Query (Heuristic Version)
1. query 입력  
   예시1 : 완전한 query  
   ```bash
    curl -s -X POST http://127.0.0.1:8750/agent/chat \
    -H 'Content-Type: application/json' \
    -d '{"message":"다운타임 장비 top10","debug":true}' | jq
    ```
   예시2 : 불완전한 query  
   ```bash
    curl -s -X POST http://127.0.0.1:8750/agent/chat \
    -H 'Content-Type: application/json' \
    -d '{"message":"장비 top10","debug":true}' | jq
    ```
2. chat 함수에서,  
   - `rid`(request_id), `thread_id`(query 입력 or uuid 생성) 등을 담은 config 생성
   - langgraph로 생성한 graph에 따라 query를 처리

3. graph invoke 과정  
   - graph.invoke()가 호출되면, 입력한 `thread_id`로 **이전 state을 로드**한다.
   - `prepare` : 현 `state`의 `current query` update
     - pending이 있는 경우 (불완전한 query가 이 thread_id를 타고 들어왔음),  
       (정상적인 추가 입력 기준) merge된 query update, pending 비움
     - pending이 없는 경우,  
       사전 규칙에 따라 query 가공 후 `current_query` update
   - `classify` : query의 요구사항 확인 (이 예제에서는 Heuristic하게 경우를 나눔)  
     - `show_sql`: 마지막 NL2SQL 결과의 SQL 원문 확인
     - `describe_schema`: SQLite schema 요약 (어떤 테이블이 있는지?)
     - `explain_last`: 마지막 NL2SQL 실행의 metadata를 요약
     - `nl2sql`: `nl2sql_router`을 통해 NL2SQL 바로 실행할지, or clarification 요구할지 판단  
       → `run_nl2sql`: 정상적인 query에 한해 nl2sql pipeline 실행

## 실행 시 포인트
- message에 질문/답변 내용이 누적되고 있음을 확인
- 최초 질문 이후, `thread_id` 유지하여 입력하면 대화 내용이 보존됨

## 실행 예시
- 불완전한 query를 전송   
  → 추가 질문을 유도하는 내용을 제공함.
```bash
curl -s -X POST http://127.0.0.1:8750/agent/chat -H 'Content-Type: application/json' -d '{"message":"장비 top10","debug":true}' | jq
{
  "ok": true,
  "request_id": "20fe3bed-5017-4929-a1ee-d4d2064970ed",
  "thread_id": "e1da7301-1ef9-40b1-9f85-752fc27ca2fd",
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

- 추가 질문에서 추천한 키워드를 제공  
  → 정상적으로 10개에 대한 정보를 제공함.
```bash
curl -s -X POST http://127.0.0.1:8750/agent/chat -H 'Content-Type: application/json' -d '{"message":"다운타임","debug":true,"thread_id":"e1da7301-1ef9-40b1-9f85-752fc27ca2fd"}' | jq
{
  "ok": true,
  "request_id": "c5834c86-5449-44bb-a02c-f7fc4e9fe3b1",
  "thread_id": "e1da7301-1ef9-40b1-9f85-752fc27ca2fd",
  "assistant_message": "다운타임 TOP 결과 (hours):\n1. CMP-074: 32.75h\n2. DIFF-183: 32.25h\n3. CMP-002: 31.117h\n4. DIFF-123: 30.65h\n5. CMP-169: 29.75h\n6. ETCH-129: 29.233h\n7. CVD-086: 29.233h\n8. CVD-189: 29.2h\n9. DIFF-118: 28.9h\n10. DIFF-038: 28.583h\n\n후속 예: `1등 장비 이벤트 보여줘`, `ETCH-01 다운타임 보여줘`",
  "intent": "nl2sql",
  "current_query": "다운타임 장비 top10",
  "pending": null,
  "last_nl2sql_ok": true,
  "last_nl2sql": {
    "ok": true,
    "request_id": "c5834c86-5449-44bb-a02c-f7fc4e9fe3b1",
    "kg_version": "8a764a11-d30c-42b3-a6f9-36582e4a1e70",
    "query_original": "다운타임 장비 top10",
    "query_rewritten": "다운타임 장비 top10",
    "rewrite_debug": {
      "tokens": [
        "다운타임",
        "장비"
      ],
      "canon_hits": 0,
      "canon_misses": 2
    },
    "kg_context": {
      "terms": [
        "다운타임",
        "장비"
      ],
      "mappings": []
    },
    "kg_context_debug": {
      "cache": "miss",
      "terms": [
        "다운타임",
        "장비"
      ]
    },
    "sql": "SELECT a.asset_name,\n               ROUND(SUM((julianday(e.end_ts) - julianday(e.start_ts)) * 24.0), 3) AS downtime_hours\n        FROM events e\n        JOIN assets a ON a.asset_id = e.asset_id\n        WHERE e.event_type = 'downtime' AND e.end_ts IS NOT NULL\n        GROUP BY a.asset_name\n        ORDER BY downtime_hours DESC\n        LIMIT 10;",
    "rows": [
      {
        "asset_name": "CMP-074",
        "downtime_hours": 32.75
      },
      ..., # 길어서 생략
      {
        "asset_name": "DIFF-038",
        "downtime_hours": 28.583
      }
    ],
    "timings_ms": {
      "rewrite": 39.117852225899696,
      "kg_context": 8.154920302331448,
      "sqlgen": 0.12304633855819702,
      "execute": 13.679382391273975,
      "total": 61.12028378993273
    },
    "cache": {
      "query_cache": "miss",
      "kgctx_cache": "miss",
      "canon_hits": 0,
      "canon_misses": 2
    }
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

- 추천된 추가 질문 반응 예시  
  → "1등 장비" 문맥을 파악하고 이벤트 50개 display
```bash
curl -s -X POST http://127.0.0.1:8750/agent/chat -H 'Content-Type: application/json' -d '{"message":"1등 장비 이벤트 보여줘","debug":true,"thread_id":"e1da7301-1ef9-40b1-9f85-752fc27ca2fd"}' | jq
{
  "ok": true,
  "request_id": "5cbe8d76-d39b-41be-b30f-f1bf03a81a93",
  "thread_id": "e1da7301-1ef9-40b1-9f85-752fc27ca2fd",
  "assistant_message": "50건의 결과를 조회했습니다. (예: 첫 행: {'event_id': 5920, 'asset_name': 'CMP-074', 'event_type': 'alert', 'start_ts': '2026-03-06 12:54:44', 'end_ts': '2026-03-06 13:17:44', 'severity': 2})",
  "intent": "nl2sql",
  "current_query": "CMP-074 이벤트",
  "pending": null,
  "last_nl2sql_ok": true,
  "last_nl2sql": {
    "ok": true,
    "request_id": "5cbe8d76-d39b-41be-b30f-f1bf03a81a93",
    "kg_version": "8a764a11-d30c-42b3-a6f9-36582e4a1e70",
    "query_original": "CMP-074 이벤트",
    "query_rewritten": "CMP-074 이벤트",
    "rewrite_debug": {
      "tokens": [
        "CMP",
        "074",
        "이벤트"
      ],
      "canon_hits": 0,
      "canon_misses": 3
    },
    "kg_context": {
      "terms": [
        "CMP",
        "074",
        "이벤트"
      ],
      "mappings": []
    },
    "kg_context_debug": {
      "cache": "miss",
      "terms": [
        "CMP",
        "074",
        "이벤트"
      ]
    },
    "sql": "SELECT e.event_id, a.asset_name, e.event_type, e.start_ts, e.end_ts, e.severity\n        FROM events e\n        JOIN assets a ON a.asset_id = e.asset_id\n        WHERE a.asset_name = 'CMP-074'\n        ORDER BY e.start_ts DESC\n        LIMIT 50;",
    "rows": [
      {
        "event_id": 5920,
        "asset_name": "CMP-074",
        "event_type": "alert",
        "start_ts": "2026-03-06 12:54:44",
        "end_ts": "2026-03-06 13:17:44",
        "severity": 2
      },
      ...,
      {
        "event_id": 5871,
        "asset_name": "CMP-074",
        "event_type": "alert",
        "start_ts": "2026-02-23 19:20:32",
        "end_ts": "2026-02-23 19:43:32",
        "severity": 1
      }
    ],
    "timings_ms": {
      "rewrite": 30.565916560590267,
      "kg_context": 7.002893835306168,
      "sqlgen": 0.07672887295484543,
      "execute": 3.111249767243862,
      "total": 40.763216093182564
    },
    "cache": {
      "query_cache": "miss",
      "kgctx_cache": "miss",
      "canon_hits": 0,
      "canon_misses": 3
    }
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
    },
    {
      "role": "human",
      "content": "1등 장비 이벤트 보여줘"
    },
    {
      "role": "ai",
      "content": "50건의 결과를 조회했습니다. (예: 첫 행: {'event_id': 5920, 'asset_name': 'CMP-074', 'event_type': 'alert', 'start_ts': '2026-03-06 12:54:44', 'end_ts': '2026-03-06 13:17:44', 'severity': 2})"
    }
  ]
}
```

