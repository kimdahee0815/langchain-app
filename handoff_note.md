# Day 3 - RAG Chain Baseline 결과

## 질문

휴가 신청 절차는 어떻게 되나요?

## 실행 결과

```json
{
  "answer": "휴가 신청 절차는 다음과 같습니다:\n\n1. 그룹웨어의 근태 관리 메뉴에서 휴가 신청서를 작성한다.\n2. 신청서에 휴가 유형(연차, 반차, 경조 휴가)과 기간을 입력한다.\n3. 결재선으로 소속 팀장을 1차 승인자로 지정한다.\n4. 휴가 시작일 기준 최소 3영업일 전에 신청해야 한다.\n\n이 절차에 따라 휴가를 신청하시면 됩니다.",
  "sources": [
    {
      "source": "docs/sample.pdf",
      "page": 2,
      "start_index": 0,
      "snippet": "제3장 휴가 신청 절차\n제8조(휴가 신청 절차) 휴가 신청은 그룹웨어의 근태 관리 메뉴에서 휴가 신청서를 작성하는\n것으로 시작한다. 신청서에는 휴가 유형(연차, 반차, 경조 휴가)과 기간을 입력하고, 결재선은\n소속 "
    },
    {
      "source": "docs/sample.pdf",
      "page": 2,
      "start_index": 168,
      "snippet": "연차 휴가는 입사일 기준으로 산정되며, 미사용 연차는 회계연도 말에 수당으로 정산하거나 다음\n해로 이월할 수 있다. 이월 가능한 연차는 최대 5일이며, 부서장 승인이 필요하다.\n제9조(승인 절차) 1차 승인자인 팀장"
    },
    {
      "source": "docs/sample.pdf",
      "page": 0,
      "start_index": 0,
      "snippet": "사내 휴가 규정 안내\n제1장 총칙\n본 규정은 임직원의 휴가 사용 절차와 기준을 정한다. 연차휴가는 입사일 기준으로 산정하며,\n신청은 휴가 시작일 3일 전까지 그룹웨어 근태관리 메뉴를 통해 제출하는 것을 원칙으로 한다"
    }
  ]
}
```

## 관찰 내용

* Retriever Top-K = 3으로 설정하여 총 3개의 근거 문서(chunk)를 검색함
* Answer는 검색된 docs를 Context로 사용하여 생성됨
* Sources는 동일한 docs에서 metadata를 추출하여 생성됨
* 검색은 1회만 수행하고, answer와 sources는 동일한 docs에서 Fan-Out 방식으로 파생됨
* page=2의 chunk가 휴가 신청 절차와 가장 직접적으로 관련된 근거로 검색됨
* page=0의 총칙 내용도 보조 근거로 함께 검색됨

## Baseline 메모

이 응답 로그는 Day 4에서 검색 품질(Query Expansion, Chunk Size, Retrieval 개선 등)을 적용한 후 동일 질문에 대한 answer와 sources 결과를 비교하기 위한 Baseline으로 사용한다.
