---
title: "PoC NL2SQL Framework - KG update evaluation"
date: 2026-02-25 09:30:00 +0900
summary: "NL2SQL PoC Evaluation"
categories: [heuristic_nl2sql, nl2sql]
---

## 추가 점검 항목
- KG 변경의 목적
  - 동의어 등록이 질의 해석/매핑 단계에 반영되어 사용자 발화의 유사 표기(`비가동시간`)를 기존 개념(`다운타임`)으로 정규화하는지 검증

- 성공 기준
  - `kg_version`이 업데이트 이전 값에서 변경됨
  - Neo4j에서 alias 관계가 정확히 `SYNONYM_OF`로 생성됨
  - 변경 전/후 동일 질문의 질의 결과가 정책에 맞게 개선됨

- 사전/사후 대조 실험
   1. 업데이트 전: `{"message":"비가동시간", "debug":true, ...}` 요청 시 결과 확인
   2. 업데이트 후: 동일 요청 재실행 후 결과 비교
   3. `actor`/`thread_id`를 동일 조건으로 고정해서 비교 일관성 확보

- 보완/트러블슈팅 체크
  - `X-API-KEY` 헤더에서 `$ADMIN_API_KEY$` 같이 오타가 있었는지 확인(실제 실행은 `$ADMIN_API_KEY` 사용)
  - `kg_version`이 바뀌었는데 반영이 안 됐다면 서비스 캐시 TTL/재시작 필요 여부 점검
  - alias/canonical이 바뀐 뒤에도 동일 alias가 다른 canonical에 중복 연결되지 않았는지 `before/after` 응답과 그래프를 함께 확인

-------

## 최초 KG version 확인
```bash
(.venv) (ds) dnmslyyd@ncia10:~/my_studies/DS/kg_nl2sql_platform_milestone_D_split/api$ curl -s http://127.0.0.1:8750/admin/kg/version -H "X-API-KEY: $ADMIN_API_KEY" | jq
{
  "kg_version": "b369b2ea-8cce-4949-97ff-e2ae085f4e99"
}
```
## KG에 synonym 추가
- (비가동시간) -(synonym of)-> (다운타임)

```bash
(.venv) (ds) dnmslyyd@ncia10:~/my_studies/DS/kg_nl2sql_platform_milestone_D_split/api$ curl -s -X POST http://127.0.0.1:8750/admin/synonym \
>   -H "X-API-KEY: $ADMIN_API_KEY$" -H "Content-Type: application/json" \
>   -d '{"alias_term_id":"T_BIGADONG","alias_text":"비가동시간","canonical_term_id":"T_DOWNTIME","canonical_text":"다운타임","reason":"demo_add","source_type":"manual"}' | jq
{
  "ok": true,
  "kg_version": "3d9eaa22-d387-49ee-af1d-4f699377624f",
  "before": {
    "alias": {
      "term_id": "T_BIGADONG",
      "text": "비가동",
      "canonical": false
    },
    "canonical": {
      "term_id": "T_DOWNTIME",
      "text": "다운타임",
      "canonical": true
    }
  },
  "after": {
    "alias": {
      "term_id": "T_BIGADONG",
      "text": "비가동시간",
      "canonical": false
    },
    "canonical": {
      "term_id": "T_DOWNTIME",
      "text": "다운타임",
      "canonical": true
    }
  }
}
```

## 변경 이후 KG version 확인
- 최초 KG version `"b369b2ea-8cce-4949-97ff-e2ae085f4e99"`에서 변경됨을 확인할 수 있다.

```bash
(.venv) (ds) dnmslyyd@ncia10:~/my_studies/DS/kg_nl2sql_platform_milestone_D_split/api$ curl -s http://127.0.0.1:8750/admin/kg/version -H "X-API-KEY: $ADMIN_API_KEY" | jq
{
  "kg_version": "3d9eaa22-d387-49ee-af1d-4f699377624f"
}
```


## KG에 잘 추가됐는지 확인
```bash
(ds) dnmslyyd@ncia10:~$ ~/opt/neo4j-current/bin/cypher-shell -a bolt://localhost:17687 -u neo4j -p '$ADMIN_API_KEY' "MATCH (a:Term {text:'비가동시간'})-[:SYNONYM_OF]->(c:Term {text:'다운타임'}) RETURN a.term_id, c.term_id;"
+-----------------------------+
| a.term_id    | c.term_id    |
+-----------------------------+
| "T_BIGADONG" | "T_DOWNTIME" |
+-----------------------------+

1 row
ready to start consuming query after 47 ms, results consumed after another 2 ms
```

