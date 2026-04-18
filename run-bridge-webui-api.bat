@echo off
@chcp 932 >nul
set PORT=8001
echo [INFO] �|�[�g %PORT% �ŋN�����܂�...

:: �����u���E�U�N���p�Ɋ��ϐ����Z�b�g
set BRIDGE_PORT=%PORT%

:: LLM�����TTS��API URL�ƃ|�[�g���`���܂��B
set LLM_API_URL=http://localhost:1234/v1
set TTS_API_URL=http://localhost:7860/

:: sync dependencies
uv sync

:: uvicorn ���N��
uv run python -m uvicorn bridge_server_api:app --host 0.0.0.0 --port %PORT% --log-level warning
