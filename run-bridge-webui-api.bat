@echo off
set PORT=8001
echo [INFO] ポート %PORT% で起動します...

:: 自動ブラウザ起動用に環境変数をセット
set BRIDGE_PORT=%PORT%

:: LLMおよびTTSのAPI URLとポートを定義します。
set LLM_API_URL=http://localhost:1234/v1
set TTS_API_URL=http://localhost:7860/

:: uvicorn を起動
uv run python -m uvicorn bridge_server_api:app --host 0.0.0.0 --port %PORT% --log-level warning
