```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 브라우저                          │
└──────────┬──────────────────────────┬───────────────────────────┘
           │ GET :5500                │ SSE/POST :8000
           ▼                          ▼
┌──────────────────┐      ┌───────────────────────────────────────┐
│  nginx           │      │  Agent Server  (FastAPI :8000)        │
│  UI 정적 파일    │      │                                       │
│  index.html      │      │  LangGraph StateGraph                 │
│  (:5500)         │      │  orchestrator → capa → material       │
└──────────────────┘      │             → quality → mold          │
                          │             → orchestrator_verdict    │
                          └────────┬──────────────┬───────────────┘
                                   │              │
                    LLM 추론 요청   │              │ MES API 요청
     POST /v1/chat/completions     │              │ (X-API-Key 헤더)
                                   ▼              ▼
                    ┌──────────────────┐  ┌──────────────────────┐
                    │  vLLM (:8080)    │  │  Security GW (:8002) │
                    │                  │  │  - API Key 인증      │
                    │  EXAONE 3.5 7.8B │  │  - 입력값 검증       │
                    │  GPU 추론        │  │  - Audit 로깅        │
                    └──────────────────┘  └──────────┬───────────┘
                                                     │ 프록시
                                                     ▼
                                          ┌──────────────────────┐
                                          │  MES Server (:8001)  │
                                          │  - /mes/production-capa
                                          │  - /mes/material-stock
                                          │  - /mes/quality-condition
                                          │  - /mes/mold-setup   │
                                          │  - /mes/order-conflict
                                          └──────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 모두 Docker 네트워크(delivery-agent_default) 안에서 통신
 컨테이너명으로 라우팅: vllm_server / security_server / mes_server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
