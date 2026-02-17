---
title: "RAG 작동 원리 (Retrieval-Augmented Generation)"
date: 2026-02-17
---

# Vector RAG 작동 원리 (Dense Vector Retrieval 기반 RAG)

> 업데이트: 2026-02-17  
> 목적: “Vector RAG가 무엇이고 어떻게 동작하는지”를 정리  

---

## 0. TL;DR

- **Vector RAG = (Vector) Retriever + Generator(LLM)**  
  질문을 받으면, 외부 지식(문서/DB)을 “의미 벡터(embedding)”로 검색해서 관련 컨텍스트를 가져오고, 그 컨텍스트를 LLM 프롬프트에 넣어 답변을 생성한다.
- Vector RAG에서 “검색”은 보통 이렇게 구현된다:
  1) 문서 조각(chunk)과 질의(query)를 같은 임베딩 모델로 벡터화  
  2) **유사도(cosine/dot/L2)** 기준으로 **top-k nearest neighbor**를 찾음(대부분 ANN 인덱스 사용)  
  3) (옵션) reranker로 재정렬해서 정밀도↑  
  4) 토큰 예산 내로 컨텍스트를 “패킹”해서 LLM에 주입
- 성능을 좌우하는 3대 레버:
  - **Chunking(검색 단위 설계)**: “정답 문장”이 chunk 내부에 들어가도록
  - **Retrieval 품질**: (dense/hybrid) + (rerank) + (query rewriting)
  - **Context packing**: “무엇을 얼마나, 어떤 순서로 넣는지”가 답변 품질을 결정

---

## 1. RAG의 기본 아이디어: “Parametric + Non-Parametric memory”

RAG의 대표적 관점은:

- **Parametric memory**: LLM 파라미터에 내재된 지식(학습으로 ‘압축’된 기억)
- **Non-parametric memory**: 외부 문서 인덱스(사내 위키/문서/DB 등)처럼 “검색 가능한” 기억
- query마다 외부 메모리에서 관련 텍스트를 가져와 LLM이 생성에 활용한다.

Vector RAG는 이 “외부 메모리”를 보통 다음 형태로 만든다:

- 문서를 chunk로 쪼갠 뒤,
- 각 chunk를 **d차원 vector**로 embed하고,
- vector index(vector DB/ANN index)에 저장해 두었다가,
- query를 같은 공간으로 embed해서 **가장 가까운 벡터(top-k)**를 가져온다.

> RAG 원 논문(2020)에는 “같은 문서를 전체 생성에 고정해서 쓰는 방식(**RAG-Sequence**)”과 “토큰별로 다른 문서를 참조할 수 있는 방식(**RAG-Token**)” 같은 변형도 비교한다.  
> 실무의 Vector RAG는 대개 “retrieve once → prompt에 넣고 generate once”로 시작해서, 필요하면 iterative/agentic으로 확장한다.

---

## 2. 구성요소(컴포넌트) 정리

### 2.1 데이터/인덱스(Index-time) 측

#### 2.1.1 Corpus + 메타데이터
- **Corpus**: 원문 문서(사내 문서, 위키, 논문, 매뉴얼, 티켓, 이메일 등)
- **메타데이터(metadata)**: 검색/필터/권한/출처 표시에 필수
  - 예: `doc_id`, `source_url`, `title`, `section`, `created_at`, `updated_at`, `owner`, `acl_groups`, `lang`, `tags`, `product`, `version` …

메타데이터는 단순 장식이 아니라, 아래를 가능하게 만든다:
- **ACL(Access Control List) 기반 필터링**: “볼 수 있는 문서만” 검색
- **스코프 제한**: 특정 제품/기간/팀 문서만 검색
- **근거(citation) 표기**: 답변에 출처 링크/문서명 표시

#### 2.1.2 전처리(정제)
Vector RAG 품질은 “입력 텍스트 품질”에 매우 민감하다.
- 깨진 줄바꿈/머리말/꼬리말 제거, 표/코드 블록 처리, 중복 제거
- PDF/HTML 변환 시 문단 구조 보존(제목/목차/섹션 헤더)
- 언어 감지 및 언어별 파이프라인 분기(옵션)

> 전처리를 대충하면 chunk는 ‘의미 단위’가 아니라 ‘노이즈 단위’가 되고, 임베딩도 그 노이즈를 그대로 학습한다.

#### 2.1.3 Chunking(검색 단위 설계)
Chunking은 “검색의 기본 단위(atomic unit)”를 정하는 작업이다. paper에서는 'chunking이 병목'이라고 할 정도.

- 검색 단위가 chunk이므로 **정답 문장(근거)이 chunk 내부에 존재**해야 top‑k로 잡히는 구조임.
- 너무 작으면: 검색은 잘 되지만 컨텍스트가 부족해 답이 빈약해짐
- 너무 크면: 임베딩이 평균화되어(semantic dilution) 검색 정밀도가 떨어지고, 토큰 예산도 낭비됨

자주 쓰는 방식
- **Heading-aware chunking**:  
  제목/소제목(heading) 기준으로 문단을 묶어 **문서 구조를 보존**한다. 보통 `section_path`(예: `2. Methods > 2.3 Retrieval`), `heading_level`, `section_id` 같은 구조 정보를 **메타데이터로 저장**해두면, 검색된 chunk의 “문맥(상위 섹션)”을 프롬프트에 함께 붙이거나, 같은 섹션의 인접 chunk를 확장해 붙이는 게 쉬워진다.  
  - 경계 품질을 위해 **인접 섹션/문단 간 overlap(예: 마지막 1~2문장 공유)** 를 둘 수 있고,  
  - 섹션이 너무 길면 섹션 내부는 다시 **token window로 쪼개되**, `chunk_index`, `start/end offset(토큰/문자)`을 저장해서 “앞/뒤 chunk 이어붙이기(stitching)”가 가능하게 만든다.

- **Token-based sliding window**:  
  텍스트를 **N 토큰 단위**로 자르고, 창을 옮길 때 stride를 두어 **overlap**을 만든다(예: `window=512`, `overlap=128` → stride=384). 구현이 단순하고 빠르지만 의미 경계를 무시할 수 있어, 문서가 구조화되어 있으면 heading을 prefix로 붙이거나(“현재 섹션 제목: …”) 섹션 단위로 먼저 나눈 뒤 window를 적용하기도 한다.  
  - 이 방식은 `chunk_index`가 곧 “윈도우 번호”가 되므로, `start_token/end_token`(또는 `start_char/end_char`) 같은 **offset 저장**이 특히 중요하다. 그래야 검색된 chunk 주변(±1~2 window)을 쉽게 확장해서 붙일 수 있다.

- **Semantic chunking**:  
  문장 단위로 쪼갠 뒤 **문장 임베딩 유사도/토픽 변화**를 이용해 의미적으로 자연스러운 덩어리로 묶는다(구현·튜닝 난이도↑). 예를 들어 “유사도 급락 지점”을 경계로 잡거나, greedy하게 문장을 누적하다가 주제 전환으로 판단되면 끊는 식이다.  
  - 그래도 LLM 컨텍스트/임베딩 최대 길이 때문에 `max_tokens` 제한은 필요하고, 경계 손실을 줄이려면 **경계 문장 몇 개를 overlap**(soft overlap)으로 공유하는 게 흔하다.  
  - 결과를 운영하려면 반드시 `chunk_index`와 원문 기준 `start/end offset`을 남겨서, (a) 인접 chunk 확장, (b) 원문 하이라이트/출처 표시, (c) rechunking 시 diff 추적이 가능하게 한다.

#### 2.1.4 Embedding 모델
- 문서 chunk와 query를 같은 모델로 임베딩하는 것이 기본
- 임베딩이 표현하는 것은 “의미/주제/관련성”이며, **정확한 사실성**을 보장하지는 않는다  
  → 그래서 reranking/filter/prompt 설계가 중요

운영 관점 체크포인트
- 임베딩 차원(dimension), 비용/속도, 다국어 성능, 도메인 적합성
- 임베딩 모델 변경 시: **전량 re-embed** 필요

#### 2.1.5 Vector Index(DB) / ANN
Vector RAG의 “검색”은 일반적으로 **근사 최근접 이웃(ANN; Approximate Nearest Neighbor)** 인덱스를 쓴다.
- top-k by Brute-force는 데이터가 커질수록 느려짐
- ANN은 “정확도(Recall) vs 지연시간(Latency) vs 메모리”를 trade-off로 조절

대표적인 Index 계열(예시)
- **HNSW**: Hierarchical Navigable Small World  
  그래프 기반. 높은 recall/낮은 latency, 메모리 사용↑, 업데이트 지원이 비교적 좋음
- **IVF**: Inverted File  
  클러스터 기반. 대규모에서 메모리 효율적, 파라미터(nlist/nprobe) 튜닝 필요
- **PQ(IVF-PQ)**: Product Quantization  
  양자화로 메모리↓, 정확도 손해 가능

> RAG 설계 관점에서 search는 DFS가 아니라 “vector 공간에서 top-k nearest neighbor 검색”이다.  
> (다만 HNSW 같은 인덱스는 내부적으로 그래프에서 greedy/best-first 탐색을 수행한다.)

#### 2.1.6 유사도(Similarity) 스코어
일반적인 선택지:
- **Cosine similarity**: normalized된 경우 사용
- **Dot product**
- **L2 distance**

metric 설정 checklist
- **임베딩 모델이 어떤 metric을 전제로 학습/권장되는지**  
- 벡터를 **정규화(normalize)** 하는지  
- 인덱스/DB가 해당 metric을 올바르게 지원하는지

#### 2.1.7 메타데이터 필터링 + 권한(ACL)
Vector 검색은 “의미 유사도”만 보므로, 다음을 별도로 보완해야 한다:
- **ACL**: 사용자/조직이 볼 수 있는 chunk만 반환
- **Scope Filter**: 제품/버전/기간/문서 타입 제한
- 구현 시 주의:
  - pre-filter(검색 전에 후보 제한) vs post-filter(검색 후 제거)
  - post-filter만 쓰면 “상위 후보가 필터로 다 날아가서 결과가 비는” 문제가 생길 수 있음

---

### 2.2 쿼리 처리(Query-time) 측

#### 2.2.1 Query understanding / 재작성(옵션)
질의 `q`를 그대로 임베딩하면 실패하는 케이스가 자주 있다.
- 약어/동의어, 도메인 용어, 복합 질문(multi-hop), 애매한 지시어(“그거”, “위 내용”)

자주 쓰는 기법
- **Query rewrite**: LLM으로 검색 친화적으로 재작성
- **Multi-query**: 여러 관점의 쿼리를 생성해 recall↑
- **Decomposition**: 복합 질문을 하위 질문으로 분해(멀티홉에 유리)
- **HyDE**(선택): “가상의 답변(가설 문서)”을 만든 뒤 그걸 임베딩해 검색(특정 도메인에서 효과적일 때가 있음)

#### 2.2.2 Retriever: Dense Vector Retrieval
- query embedding으로 vector index에서 top-k chunk를 가져온다.
- top-k 선택은 trade-off:
  - `K`가 너무 작으면 recall↓
  - `K`가 너무 크면 노이즈↑ + 비용↑ + context packing 난이도↑
    - context packing: top-k retrieval 결과를 LLM 컨텍스트 길이 안에 들어가도록,  
      “선택·정렬·중복제거·문맥확장·요약·포맷팅”해서 최종 프롬프트 컨텍스트를 구성하는 과정

#### 2.2.3 Hybrid retrieval(옵션)
Dense embedding은 기본적으로 “의미”를 압축해서 표현하므로, 다음과 같은 문제가 생길 수 있음.
- **희귀 토큰/정확 문자열을 과소평가**  
에러코드(E11000, ORA-00942), API 명(GetObject), 버전(v1.2.3), 파라미터 키(--max_tokens) 같은 건 의미적으로 “비슷한 설명”이 많아도, 그 토큰이 정확히 포함된 chunk를 강하게 띄우지 못할 수 있음.
- **숫자/식별자(IDs) 매칭이 불안정**  
12345와 12346은 의미상 거의 같게 보이거나, embedding space에서 구분이 약해질 수 있음.
- **토크나이저/정규화 이슈**  
하이픈, 언더스코어, 경로(/usr/bin), 스택트레이스 포맷 등이 임베딩 모델에선 noise로 취급될 때가 있음.

그래서 문자열 일치에 강한 **BM25(lexical) + dense(semantic)**를 결합하기도 한다.
- **BM25(lexical)**: "정확히 그 단어/코드/ID가 들어있는 문서"를 매우 잘 찾음 → precision/anchor 역할
- **Dense(semantic)**: 동의어/서술 표현이 달라도 의미가 같으면 찾음 → recall/coverage 역할
- 단순 가중합, 또는 **RRF(Reciprocal Rank Fusion)** 같은 랭킹 결합 사용
- 실무 경험상 hybrid는 도메인에 따라 체감 성능을 크게 올릴 수 있음

#### 2.2.4 Reranker(옵션)
bi-encoder 기반 vector 검색은 recall(놓치지 않기)을 올리기는 쉽지만, precision(맞는 근거만 올리기)에는 한계가 있다.  
→ top-k 후보를 가져온 뒤, 더 비싼 모델로 정밀하게 압축.

1. **Cross-encoder reranker**: (query, chunk)를 하나로 붙여 Transformer에 넣고, `[CLS]` 같은 요약 표현으로 relevance 점수를 직접 예측  
  - **장점**: query와 chunk를 동시에 보면서 토큰 단위 상호작용을 학습/활용  
  → “쿼리의 핵심 조건을 만족하는지” 판단이 매우 강함 (예: 에러코드/파라미터명/부정 조건/예외 조항)
  - **단점**: 후보마다 모델을 한 번씩 돌려야 해서 비용이 선형으로 증가
2. **LLM reranker**: LLM에게 후보들을 보여주고 “어떤 후보가 질문에 가장 답이 되는가?”를 요약/비교시키는 방식
  - 주요 패턴
    - **Pointwise**: 후보마다 “관련 점수 0~5”처럼 평가 → 점수로 정렬
    - **Pairwise**: A vs B 비교를 반복 (비용 큼)
    - **Listwise**: 후보 리스트를 한 번에 주고 상위 n개 ID만 고르게 함
  - 장점: 단순 relevance뿐 아니라 “answerability” 기준을 넣기 쉽다.  
    예: “이 chunk만으로 답을 쓸 수 있는가?”, “근거로 인용하기 적절한가?”, “최신 버전인가?” 같은 규칙을 프롬프트로 강제 가능
  - 단점
    - 비용/지연이 큼(특히 후보 텍스트가 길면)
    - LLM 특유의 편향(풍부한 서술, keyword, prompt injection, ...)에 의한 평가 안정성 문제
  - 보완
    - 선택 + 근거 인용 강제
    - 출력 형식을 강하게 제한: `{"top_ids":[...], "evidence":[{"id":..,"quote":"..."}]}`
    - 후보 텍스트를 “데이터로만 취급”하게 하는 방어
    - rerank 전에 후보를 표준화/압축
    - hybrid reranking (dense top-k → cross-encoder top-m → LLM reranker final top-n)

일반적인 패턴:
- 1차: vector `top_k=50~200` (빠르게 많이 가져옴)
- 2차: rerank `top_n=5~20` (정밀하게 압축)

#### 2.2.5 Context packing / Token budget
LLM의 context window는 제한되어 있으므로, top-k chunk를 그대로 다 넣을 수 없거나 넣어도 품질이 떨어짐.  
packing 단계에서 하는 일:
1. 선택(selection)
   - top-k 후보 중 실제로 넣을 chunk를 더 줄이기(예: top-k → top-n)
   - reranker 점수, MMR(Maximum Marginal Relevance search), filter(문서 중복 제거) 등
2. 정렬(ordering)
   - 중요한 chunk를 앞에 배치(LLM이 앞부분을 더 잘 쓰는 경향이 있으므로)
   - 같은 문서/섹션끼리 묶어서 문맥이 이어지게 배치
3. 중복 제거(deduplication)
   - 비슷한 chunk가 여러 개 들어오면 정보량이 늘지 않으므로 near-duplicate 제거, 동일 문장/표 반복 제거
4. 문맥 확장(stitching / window expansion)
   - 검색된 chunk 하나만으로는 문맥이 끊길 때 `chunk_index`/`offset`을 이용해 앞/뒤 인접 chunk를 붙여 “하나의 근거 블록”으로 만듦  
  (이때 토큰 예산을 넘으면 일부만 붙이거나 요약)
5. 압축/요약(compression)
   - token 예산이 부족하면 chunk의 핵심 문장만 추출하거나(문장 선택), chunk 요약만 넣기
6. 포맷팅(formatting)
   - 프롬프트에 넣기 좋은 형태로 변환: `[Source 3 | doc=... | section=... | chunk=12] ...본문...`
   - citation이나 metadata(제목/날짜/버전)를 함께 넣어 grounding 강화


#### 2.2.6 Generator(LLM) + Grounding
LLM은 컨텍스트를 참고해 답을 생성하지만, 컨텍스트 밖을 “그럴듯하게” 말할 수 있다.
그래서 프롬프트/후처리로 grounding을 강화한다.

- “근거 없으면 모른다고 말하기”
- “답변의 각 주장에 대해 근거 chunk를 붙이기”(citation)
- “컨텍스트에 없는 사실은 생성하지 않기”를 시스템/개발자 지시로 강제

#### 2.2.7 (옵션) Citation / Provenance
Chunk에 `doc_id`/`source_url`을 저장해 두면:
- 답변에서 “어떤 문서/섹션을 근거로 썼는지”를 표시 가능
- 디버깅도 쉬워짐(“이 답이 왜 나왔지?”)

---

## 3. 동작 흐름(파이프라인)

### 3.1 Index-time(사전 준비)
1. 문서 수집/정제(추출 품질 확보)  
2. 문서 chunking + metadata 부착  
3. chunk embedding 생성  
4. vector index에 저장(metadata 포함: `doc_id`, `chunk_id`, `offset`, `title`, `acl` 등)  
5. (운영 시) 문서 update/deletion을 인덱스에 반영(upsert/delete), 버전 관리

> 운영 포인트: embedding model을 바꾸거나 chunking 정책을 바꾸면 “re-embedding + re-indexing”이 사실상 필수다.

### 3.2 Query-time(실제 질의 처리)
1. 입력 query `q`  
2. (옵션) query 재작성/확장(multi-query, decomposition 등)  
3. embed `q` → vector top-k 검색(+filter/ACL)  
4. (옵션) hybrid 결합(BM25 등)  
5. (옵션) rerank로 후보 재정렬  
6. **토큰 예산 B 내로 컨텍스트 구성(pack/truncate/stitch)**  
7. LLM에 `{instruction + q + context}`를 넣고 답변 생성  
8. (옵션) 근거 표기, 후처리(형식화/검증)

#### 의사코드(pseudo-code)
```python
# ---------------------------
# index-time
# ---------------------------
chunks = chunk(docs)  # heading-aware / sliding window / semantic
vecs = embed(chunks)
index.upsert(
  vectors=vecs,
  metadatas={
    "doc_id": ...,
    "chunk_id": ...,
    "title": ...,
    "section": ...,
    "offset": ...,
    "source_url": ...,
    "acl_groups": ...,
    "updated_at": ...,
  }
)

# ---------------------------
# query-time
# ---------------------------
q = user_query

q_list = rewrite_or_expand(q)  # optional: multi-query / decomposition / HyDE
cand_pool = []

for qi in q_list:
    q_vec = embed(qi)
    cands = index.search(
        q_vec,
        top_k=K,
        filter={"acl_groups": user_groups, "product": target_product}  # optional
    )
    cand_pool.extend(cands)

cands = fuse_or_dedup(cand_pool)         # optional: RRF / dedup by doc_id
cands = rerank(q, cands)                 # optional: cross-encoder reranker
context = pack_to_budget(cands, B)       # select + stitch + truncate/summarize
answer = llm.generate(prompt=build_prompt(q, context), citations=True)
return answer
```

#### 다이어그램(mermaid)
<div class="mermaid">
flowchart TB
  subgraph Indexing
    D["Documents"] --> C["Chunking"]
    C --> E["Embeddings"]
    E --> V["Vector Index/DB"]
  end

  subgraph Query
    Q["User Query"] --> QE["Query Embedding"]
    QE --> R["Top-k Retrieval"]
    R --> RR["Rerank (optional)"]
    RR --> P["Context Packing<br/>Token Budget"]
    P --> L["LLM Generation"]
    L --> A["Answer<br/>(optional) citations"]
  end

  V --> R
</div>

---

## 4. Why Token budget Matters?

RAG는 보통 LLM context window가 제한돼서, **모든 관련 문서를 다 넣지 못한다.**  
그래서 실전에서는 아래가 품질을 좌우한다:

- **B(총 입력 컨텍스트 토큰 예산)** 안에서 (=지시문/질문/히스토리/근거문서/출력 여유분 포함)
   - 어떤 chunk를 넣을지(선택)
   - 어떤 순서로 넣을지(정렬)
   - 중복을 얼마나 줄일지(커버리지/다양성)
   - 긴 chunk를 어떻게 자를지(truncation)
   - 필요하면 어떻게 압축할지(요약/발췌)

**자주 쓰는 packing 전략**
1. rerank 후 상위 chunk부터 넣기
2. 문서 다양성 확보(MMR)
3. 같은 문서에서 **이웃 chunk를 함께** 넣어 문맥 보강(stitching; `chunk_index`/`offset` 기반)
4. 표/코드/정의는 원문 유지, 서술은 요약(도메인에 따라 반대일 수도 있음)

---

## 5. Variants

- **Naive Vector RAG**: retrieve once + generate once (가장 기본)
- **Multi-query / query expansion**: 여러 쿼리로 검색해서 recall↑
- **Iterative(Agentic) RAG**: 검색→읽기→새 쿼리→재검색 반복(멀티홉/복합 질문에 유리)
- **Hybrid RAG**: BM25(lexical) + embedding(dense) 결합(코드/식별자/정확한 키워드에 강함)
- **Reranking 강화**: cross-encoder로 top-k 재정렬(정밀도↑)
- **Long-context RAG**: 컨텍스트 윈도우가 큰 모델로 더 많이 넣되, 비용 증가 가능
- **Router/Ensemble**(옵션): 질문 유형에 따라 retriever/인덱스를 선택(FAQ/정책/코드/로그 등)

---

## 6. 실패 모드(Failure modes) & 체크리스트

### 6.1 Ingestion/Chunking 실패
- 문서 추출(PDF/HTML) 품질이 낮아 문장/표 구조가 깨짐
- chunk 경계가 잘못되어 “정답 문장”이 쪼개져 버림
- 메타데이터(제목/섹션/ACL)가 누락되어 packing/citation/필터가 무력화됨

**대응**: 추출 개선, heading-aware chunking, overlap 조정, parent-child 구조 저장

### 6.2 Retrieval 실패
- 임베딩이 도메인 용어/약어를 잘 못 잡음
- dense가 약한 영역(에러 코드/정확 매칭)에서 miss
- top-k/ANN 파라미터가 부적절(너무 aggressive한 근사로 recall↓)

**대응**: hybrid search, query rewriting, K 튜닝, ANN 파라미터 튜닝(예: HNSW efSearch)

### 6.3 Rerank/Selection 실패
- 후보는 가져왔지만 reranker가 잘못 고름
- 중복 chunk가 상위를 독식해 다양성이 사라짐
- 하나의 문서만 과도하게 들어가서 편향된 컨텍스트가 됨

**대응**: rerank 모델/프롬프트 튜닝, MMR, doc-level dedup, 섹션 다양성 제약

### 6.4 Generation 실패(환각/과장/누락)
- 컨텍스트에 없는 내용을 LLM이 만들어냄
- 컨텍스트가 많아도 “핵심 근거”를 못 잡고 요약이 흐림
- 질문과 무관한 부분을 길게 답함

**대응**: 인용 강제(prompt), “컨텍스트 밖 금지” 규칙, 답변 구조화(요약→근거→결론), 검증 단계(judge) 추가

---

## 7. 평가(Evaluation) & 디버깅(실전)

### 7.1 Retriever 평가(검색 품질)
- 목적: “정답이 들어있는 chunk를 top-k 안에 넣는가?”
- 자주 쓰는 지표:
  - **Recall@k**: 정답 chunk가 top-k에 포함되는 비율
  - **MRR**: 정답이 얼마나 위에 랭크되는지(순위 민감)
  - **nDCG**: 관련도 등급이 있을 때 유용

### 7.2 Generator 평가(답변 품질)
- **Answer correctness**(정확성)
- **Faithfulness / Groundedness**: 컨텍스트에 근거한 주장만 하는지
- **Citation accuracy**: 인용이 실제 근거를 가리키는지
- **Helpfulness / format compliance**

### 7.3 디버깅 플레이북(빠르게 원인 찾기)
1) “정답이 corpus에 존재하는가?” (없으면 RAG로도 못 푼다)
2) “정답이 chunk 내부에 들어가 있는가?” (chunking/추출 문제)
3) “retrieval top-k에 들어오는가?” (임베딩/하이브리드/K/ANN 문제)
4) “rerank 후에도 살아남는가?” (reranker 문제)
5) “packing 후에도 컨텍스트에 포함되는가?” (토큰 예산/중복 문제)
6) “LLM이 컨텍스트를 사용했는가?” (프롬프트/지시 문제)

---

## 8. 실무 기본 세팅(시작점)

- Chunking:
  - heading-aware + 문단 기반 + 적당한 overlap
  - `doc_id / title / section / chunk_index / offset / source_url / acl` 메타데이터 저장
- Retrieval:
  - dense top_k는 작게 시작하지 말고(예: 50~200 후보)
  - rerank로 5~20으로 압축하는 2-stage를 권장
  - 에러 코드/정확 매칭이 많으면 hybrid 고려
- Packing:
  - 중복 제거 + doc 다양성 확보(MMR)
  - 이웃 chunk stitching(전후 1개)로 문맥 보강
- Prompt:
  - “근거 없으면 모른다고 말하기”
  - “가능하면 인용 표시” (디버깅/신뢰성에 도움)

---

## 9. 참고 문헌/링크

- [RAG 원 논문 (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [NeurIPS 2020 PDF](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)