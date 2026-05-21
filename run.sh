#!/bin/bash
# 로컬 개발용 (Docker 없이 직접 실행할 때)

echo "=== MES Mock 서버 (포트 8001) 시작 ==="
cd server2_mes && uvicorn main:app --host 0.0.0.0 --port 8001 &
MES_PID=$!

sleep 2

echo "=== Agent 서버 (포트 8000) 시작 ==="
cd ../server1_agent && uvicorn main:app --host 0.0.0.0 --port 8000 &
AGENT_PID=$!

echo ""
echo "서버 실행 중..."
echo "  MES Mock : http://localhost:8001/docs"
echo "  Agent    : http://localhost:8000/docs"
echo "  UI       : ui/index.html 을 브라우저로 직접 열거나 python -m http.server 5500 으로 서빙"
echo ""
echo "종료: Ctrl+C"

trap "kill $MES_PID $AGENT_PID" EXIT
wait
