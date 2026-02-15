---
title: "RAG 작동 원리 (Retrieval-Augmented Generation)"
date: 2026-02-15
---

# RAG 작동 원리 (Retrieval-Augmented Generation)

> 업데이트: 2026-02-15  
> 목적: “RAG가 무엇이고 어떻게 동작하는지”를 **구현 관점**에서 한 번에 정리

---

## 0. TL;DR

- **RAG = Retriever + Generator(LLM)**  
  질문을 받으면, 외부 지식(문서/DB)을 **검색(retrieval)** 해서 관련 컨텍스트를 찾고, 그 컨텍스트를 LLM 프롬프트에 넣어 **답변을 생성(generation)** 한다.
- 핵심 장점:
  - 모델 파라미터 밖 지식을 활용 → **도메인 지식 보강**
  - 최신/사내 문서처럼 모델이 모르는 정보 활용 가능
  - (설계하면) 근거(출처) 제공 가능
- 핵심 병목:
  - Retrieval이 틀리면 답도 틀림(“Garbage In, Garbage Out”)
  - 컨텍스트 예산(토큰 제한) 때문에 “무엇을 얼마나 넣을지”가 품질을 좌우

---

## 1. RAG의 기본 아이디어: “파라메트릭 + 비파라메트릭 메모리”

RAG의 대표적 정의(원 논문 관점)는:

- **Parametric memory**: LLM 파라미터에 내재된 지식
- **Non-parametric memory**: 외부 문서 인덱스(예: Wikipedia/사내 문서) 같은 “검색 가능한 메모리”
- 질의마다 외부 메모리에서 관련 텍스트를 가져와 LLM이 생성에 활용한다.

> 참고: RAG 원 논문은 “같은 문서를 전체 생성에 고정해서 쓰는 방식”과 “토큰별로 다른 문서를 참조할 수 있는 방식” 같은 변형도 비교한다.

---

## 2. 구성요소(컴포넌트) 정리

### 2.1 데이터/인덱스 측
- **Corpus**: 원문 문서(사내 문서, 위키, 논문 등)
- **Chunking**: 문서를 “검색 단위”로 쪼갠 것(문단/문장/토큰 기준)
- **Embedding 모델**: 텍스트를 벡터로 변환
- **Vector index(DB)**: 벡터를 저장하고 top-k 검색을 빠르게 수행(ANN)

> 흔히 “search가 DFS냐?” 같은 질문이 나오는데, **RAG 설계 관점에서 search는 DFS가 아니라 ‘top-k nearest neighbor 검색’**이다.  
> (다만 내부적으로 HNSW 같은 인덱스는 그래프 위 best-first/greedy 탐색을 쓰기도 함)

### 2.2 쿼리 처리 측
- **Query understanding**: 질의 재작성/확장(옵션)
- **Retriever**: top-k chunk를 가져옴(semantic search / hybrid search)
- **Reranker**(옵션): cross-encoder로 top-k를 재정렬(정밀도↑)
- **Context packing**: 토큰 예산 안에서 문서를 “얼마나/어떻게” 넣을지 결정
- **Generator(LLM)**: 컨텍스트 기반 답변 생성
- **Citations/Provenance**(옵션): 어떤 chunk를 근거로 썼는지 표시

---

## 3. 동작 흐름(파이프라인)

### 3.1 Index-time(사전 준비)
1) 문서 수집/정제
2) 문서 청킹(chunking)
3) 각 chunk 임베딩 생성
4) 벡터 인덱스에 저장(메타데이터 포함: doc_id, chunk_id, 위치, 원문 링크 등)

### 3.2 Query-time(실제 질의 처리)
1) 입력 질의 `q`
2) (옵션) 질의 재작성/확장(예: multi-query)
3) `q`를 임베딩 → top-k 검색
4) (옵션) rerank
5) **토큰 예산 B 내로 컨텍스트 구성(pack/truncate)**
6) LLM에 `{q + context}`를 넣고 답변 생성
7) (옵션) 근거 표기, 후처리(형식화/검증)

#### 의사코드(pseudo-code)
```python
# index-time
chunks = chunk(docs)
vecs = embed(chunks)
index.add(vecs, meta={"chunk_text": ..., "doc_id": ..., "offset": ...})

# query-time
q_vec = embed(query)
cands = index.search(q_vec, top_k=K)
cands = rerank(query, cands)  # optional
context = pack_to_budget(cands, budget_tokens=B)
answer = llm.generate(prompt=query + "\n\n" + context)
return answer
```

#### 다이어그램(mermaid)
```mermaid
flowchart LR
  subgraph Indexing
    D[Documents] --> C[Chunking]
    C --> E[Embeddings]
    E --> V[Vector Index/DB]
  end

  subgraph Query
    Q[User Query] --> QE[Query Embedding]
    QE --> R[Top-k Retrieval]
    R --> RR[Rerank (optional)]
    RR --> P[Context Packing / Token Budget]
    P --> L[LLM Generation]
    L --> A[Answer + (optional) citations]
  end

  V --> R
```

---

## 4. “Token budget(예산)”이 왜 중요하나?

RAG는 보통 LLM 컨텍스트 윈도우가 제한돼서, **모든 관련 문서를 다 넣지 못한다.**  
그래서 실전에서는 다음이 핵심 설계 포인트다:

- **B(컨텍스트 토큰 예산)** 안에서
  - 어떤 chunk를 넣을지(선택)
  - 어떤 순서로 넣을지(정렬)
  - 중복을 얼마나 줄일지(커버리지/다양성)
  - 긴 chunk를 어떻게 자를지(truncation)
- 이 “선택/패킹”이 나쁘면 **retrieval이 맞았어도** 답이 흔들린다.

---

## 5. 자주 쓰는 변형(Variants)

- **Naive RAG**: retrieve once + generate once
- **Multi-query / query expansion**: 여러 쿼리로 검색해서 recall↑
- **Iterative(Agentic) RAG**: 검색→읽기→새 쿼리→재검색을 여러 번 반복(멀티홉/복합 질문에 유리)
- **Hybrid RAG**: BM25(lexical) + embedding(dense) 결합
- **Reranking 강화**: cross-encoder로 top-k 재정렬
- **Long-context RAG**: 컨텍스트 윈도우가 큰 모델로 더 많이 넣되, 비용 증가 가능

---

## 6. 실패 모드(Failure modes) & 체크리스트

### 6.1 Retrieval 실패
- chunking이 잘못되어 “정답 문장”이 조각나거나
- 용어/약어/도메인 표현이 임베딩에 반영이 안 되거나
- top-k가 너무 작거나/너무 크거나

**대응**: chunking/overlap 튜닝, hybrid search, rerank, query rewriting

### 6.2 Generation 실패(환각/과장)
- 컨텍스트에 없는 내용을 LLM이 만들어냄
- 컨텍스트가 많아도 “핵심 근거”를 못 잡고 요약이 흐림

**대응**: 인용 강제(prompt), answer grounding 규칙, judge/검증 단계(옵션)

---

## 7. 참고 문헌/링크

- RAG 원 논문 (Lewis et al., 2020):  
  https://arxiv.org/abs/2005.11401  
  (NeurIPS 2020 PDF) https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
