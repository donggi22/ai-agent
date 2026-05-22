import os

VLLM_BASE_URL     = os.getenv("VLLM_BASE_URL",     "http://localhost:8080/v1")
MES_BASE_URL      = os.getenv("MES_BASE_URL",      "http://localhost:8001")
SECURITY_BASE_URL = os.getenv("SECURITY_BASE_URL", "http://localhost:8002")
GATEWAY_API_KEY   = os.getenv("GATEWAY_API_KEY",   "agent-secret-key-1234")
MODEL_NAME        = os.getenv("MODEL_NAME",        "exaone")

# vLLM은 OpenAI-compatible이라 api_key 불필요, 더미값 넣음
OPENAI_API_KEY = "dummy"
