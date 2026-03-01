---
title: "PoC NL2SQL Framework - Evaluation"
date: 2026-02-25 09:15:00 +0900
summary: "NL2SQL PoC Evaluation"
categories: [heuristic_nl2sql, nl2sql]
---

## 평가 과정 요약
```bash
cd kg_nl2sql_platform_milestone_D_split/api
python tools/eval_milestone_d.py --scenario scale --host 127.0.0.1 --port 8750
```
기본 suite (평가 케이스 모음 파일): `eval/suites/scale_augmented.jsonl`  
실행 기준 DB: `data/app_large.db`  
요약 저장: `runs/<timestamp>_<run_id>/summary.json`, `report.md`, `results.jsonl`

## 1) 수행한 평가 항목
- 기본 상호작용/서비스 상태 체크
  - `health` 요청 응답 확인 (`/health`, `status=ok`)
  - `/chat/query`, `/agent/chat` 경로에서 쿼리 결과와 요청/응답 일관성 검사
- 데이터량 검증
  - `db_stats` 체크에서 최소값 충족 여부 확인 (`assets/events/metrics`)
- 쿼리 기능/정합성
  - `정지시간 많은 장비 top10` 1회 호출
  - `온도 높은 장비 top10` 1회 호출
  - 각 쿼리는 최소 행수(`min_rows`) 충족 여부를 검사하고, 응답 `status=200` 및 요청/응답 request-id 정합성을 확인
  - 캐시 미스(non-hit) 케이스는 audit 추적(`audit_logs`) 존재도 함께 검증
- 캐시 동작
  - `정지시간 많은 장비 top10`를 5회 반복(`chat_query_repeat`)
  - 캐시 활성화 시 적어도 1회 이상 `query_cache=hit` 기대
- 배치 확장성/성능
  - `asset_batch` 타입으로 자산 20개를 랜덤 샘플링
    - `events` 쿼리: `"<asset> 이벤트"`
    - `downtime` 쿼리: `"<asset> 다운타임"`(다운타임 이벤트 존재 자산 기준)
  - 각 자산별 쿼리가 최소 행수 조건(`min_rows`)을 통과하는지 확인
  - 요청 지연시간을 수집해 p50/p95 산출
- 메트릭 기반 집계
  - 실행 전/후의 `/metrics` 카운터 차이 측정
    - `http_requests_total`
    - `nl2sql_requests_total`
    - `nl2sql_cache_hit_total`

## 2) 평가 결과

- 환경/입력
  - scenario: `scale`
  - sqlite_db: `/data/home/dnmslyyd/my_studies/DS/kg_nl2sql_platform_milestone_D_split/data/app_large.db`
  - asset/events/metrics: 각각 200 / 16000 / 40000
- 정량 결과
  - 전체 체크: `passed=51`, `failed=0`
  - 실행 시간: `2.35s`
  - DB 통계 (`db_stats` 체크 시): `assets=200`, `events=16000`, `metrics=40000`, `audit_logs=0`
- Audit/메트릭
  - 실행 전/후 `audit_diff`: `43` (실행 중 43건의 audit 로그 생성)
  - metrics diff:
    - `http_requests_total`: `+49`
    - `nl2sql_requests_total`: `+43`
    - `nl2sql_cache_hit_total`: `+4`
- 성능
  - `latency_chat_p50_ms`: `29.78ms`
  - `latency_chat_p95_ms`: `73.38ms`
  - `cache_hits_observed`: `4`
- 해석
  - **평가 전 구간 기능은 모두 통과** (FAIL 0)  
  - scale 데이터셋(200 assets)에 대해 건강성, 캐시 동작, 다중 자산 질의, 지연시간 특성까지 기준치 충족  
  - 캐시 설정이 활성화된 상태에서 반복 질의 캐시가 실제로 동작(총 4회 hit)했으며, 대체로 50ms 내외의 빠른 응답 안정성 확인

## 참고: 평가 로직 요약

- `chat_query_repeat`는 5번 반복 질의 중 최소 1회 이상 cache hit을 기대한다.
- `asset_batch`는 자산별 쿼리를 각각 개별 체크로 분해해 집계한다.

## LLM으로 교체 대상
1. query preparation: 현재는 고정된 형태(예. 온도 장비 top1) input만 처리가 가능
2. query classification 
3. nl2sql router
4. nl2sql template
