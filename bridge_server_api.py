"""
bridge_server.py  –  WebUI版 bridge3.py
依存: fastapi uvicorn openai python-multipart

起動:
    uvicorn bridge_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os, subprocess, time, re, threading, queue, asyncio, json, zipfile, shutil, urllib.parse, tempfile, uuid, copy
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, model_validator
from gradio_client import Client, handle_file
from PIL import Image
from game_mode import GameEngine, GameError
from game_mode.llm import OpenAIGameLLM
from game_mode.transcript import GameTranscriptWriter
import io
import math
import webbrowser
import base64
import urllib.request as _urllib_req
import urllib.error as _urllib_err

# --- 設定ファイル ---
SETTINGS_FILE = "bridge_settings.json"
TTS_CONFIG_FILE = "tts_config.json"   # TTS詳細設定の外部ファイル

PROMPTS_DIR = "prompts"

DEFAULT_SETTINGS = {
    "tts_batch_size": 0,      # TTSに投げる文章数 (0=すべて)
    "dark_mode": False,
    "blur_mode": True,
    "tts_enabled": True,
    "bg_enabled": True,
    # システムプロンプト設定
    "system_prompt_file": "base_system_prompt.txt",   # 初回デフォルト
    "system_prompt_after_char": False,  # True=キャラクタープロンプトの後ろにシステムプロンプトを配置
    "user_slots": ["", "", "", "", ""],  # ユーザー編集スロット x5
    "active_user_slot": -1,             # 使用中のユーザースロット (-1=なし)
    "slot_and_file": False,             # True=スロットとファイルを同時適用
    # ダウンロード設定
    "ffmpeg_to_mp3": False,        # True=ダウンロード時にMP3変換
    "ffmpeg_path_enabled": False,  # True=ffmpegパスを手動指定
    "ffmpeg_path": "",             # ffmpeg実行ファイルのパス
    # 表示設定
    "bubble_alpha": 0.25,          # 吹き出し背景の不透明度 (0.0〜1.0)
    "text_shadow": False,          # True=メッセージテキストにシャドウ付与
    "bubble_blur": True,           # True=吹き出し背景にぼかしエフェクト
    # キャラクター読み出し設定
    "char_dir_also_default": False, # True=CHAR_DIR設定時もデフォルトフォルダを追加で読み込む
    # LLM接続設定（永続化）
    "llm_use_env":  False,         # True=環境変数LLM_API_URLを優先使用
    "llm_preset":   "lm-studio",   # 選択中のプリセットID
    "llm_host":     "localhost",   # LLMサーバのホスト/IP
    "llm_port":     1234,          # LLMサーバのポート
    "llm_path":     "/v1",         # APIパス
    "llm_api_key":  "lm-studio",   # APIキー (ローカルサーバ用ダミーキー等)
    "llm_model":    "",            # モデル名 (空="local-model"フォールバック)
    # TTS接続設定（永続化）
    "tts_use_env":  False,         # True=環境変数TTS_API_URLを優先使用
    "tts_host":     "localhost",   # TTSサーバのホスト/IP
    "tts_port":     7860,          # TTSサーバのポート
    # 参照音声なしキャラのTTS動作
    "no_voice_mode": "off",        # "off"=音声オフで使用 / "default_voice"=デフォルト音声で使用
    "tts_api_version": "auto",     # "auto"=自動検出 / "v2" / "v3" / "v4"
    # テキスト読み上げ（ファイル/ペースト）設定
    "tts_file_batch_size": 2,      # 読み上げ1吹き出しあたりの文章数 (1〜5)
    "tts_read_strip_enabled": True,    # 読み上げ時に除去文字を適用するか
    "tts_read_strip_chars":   "()（）",  # 除去する括弧ペア文字列
    "tts_read_max_chars":     1024,    # TTS送出1回あたりの最大文字数
    # 自動応答設定
    "auto_reply_delay":   3,       # 自動応答待機秒数 (1〜30)
    "auto_reply_trigger": "audio", # "audio"=音声完了後 / "llm"=テキスト生成完了後
    "auto_reply_user_prompt": "",  # 空=デフォルトプロンプト使用
    # Speaker Embedding設定
    "prefer_speaker_embedding": True,   # True=Embeddingを優先(両方ある場合)
    # 記憶・関係性設定
    "memory_enabled":         True,  # True=キャラごとにユーザーを記憶する
    "memory_update_interval": 6,     # ユーザー発言N回ごとに記憶を更新 (1〜30)
    # 会話履歴設定
    "history_max_messages":   30,    # LLMに渡す履歴の最大メッセージ数 (0=無制限)
}

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            merged = {**DEFAULT_SETTINGS, **saved}
            return merged
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)

_settings_lock = threading.Lock()

def save_settings(data: dict):
    """アトミック書き込み: 一時ファイルに書いてからリネームすることで
    部分書き込みによる JSON 破損を防ぐ。"""
    dir_name = os.path.dirname(os.path.abspath(SETTINGS_FILE))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp', prefix='settings_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, SETTINGS_FILE)  # プラットフォーム問わずアトミック
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise

# --- 設定 ---
# 環境変数（起動時参照用）
_ENV_LLM_API_URL = os.getenv("LLM_API_URL", "")
_ENV_TTS_API_URL = os.getenv("TTS_API_URL", "")

# LLMサーバプリセット
LLM_PRESETS = [
    {"id": "lm-studio",      "label": "LM Studio",            "host": "localhost", "port": 1234,  "path": "/v1", "key": "lm-studio", "supports_unload": True},
    {"id": "ollama",         "label": "Ollama",                "host": "localhost", "port": 11434, "path": "/v1", "key": "ollama",    "supports_unload": True},
    {"id": "text-gen-webui", "label": "text-generation-webui", "host": "localhost", "port": 5000,  "path": "/v1", "key": "",          "supports_unload": True},
    {"id": "koboldcpp",      "label": "KoboldCpp",             "host": "localhost", "port": 5001,  "path": "/v1", "key": "",          "supports_unload": False},
    {"id": "llama-server",   "label": "llama.cpp server",      "host": "localhost", "port": 8080,  "path": "/v1", "key": "none",      "supports_unload": False},
    {"id": "custom",         "label": "カスタム",              "host": "",          "port": None,  "path": "/v1", "key": "",          "supports_unload": False},
]


def _build_llm_url(settings: dict) -> str:
    """設定からLLM API URLを構築する。llm_use_env=True かつ env var があれば環境変数を優先。"""
    if settings.get("llm_use_env") and _ENV_LLM_API_URL:
        return _ENV_LLM_API_URL
    host = (settings.get("llm_host") or "localhost").strip()
    port = settings.get("llm_port") or 1234
    path = (settings.get("llm_path") or "/v1").strip()
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def _build_tts_url(settings: dict) -> str:
    """設定からTTS API URLを構築する。tts_use_env=True かつ env var があれば環境変数を優先。"""
    if settings.get("tts_use_env") and _ENV_TTS_API_URL:
        return _ENV_TTS_API_URL
    host = (settings.get("tts_host") or "localhost").strip()
    port = settings.get("tts_port") or 7860
    return f"http://{host}:{port}/"


# ── 起動時に設定ファイルから接続情報を読み込む ──────────────────────────────
_startup_settings = load_settings()
LLM_API_URL = _build_llm_url(_startup_settings)
LLM_API_KEY = (_startup_settings.get("llm_api_key") or "lm-studio")
LLM_MODEL   = (_startup_settings.get("llm_model") or "")
TTS_API_URL = _build_tts_url(_startup_settings)

# その他の設定
CHAR_DIR_ENV     = os.getenv("CHAR_DIR")                               # None = 未設定
CHAR_DIR         = CHAR_DIR_ENV if CHAR_DIR_ENV else "characters"
DEFAULT_CHAR_DIR = "characters"                                         # デフォルトフォルダ
BASE_PROMPT_FILE = os.path.join(PROMPTS_DIR, "base_system_prompt.txt")
MAX_HISTORY     = 10
WINDOW_WIDTH    = 300


# ── TTS詳細設定 (tts_config.json から読み込み) ────────────────────────────
# v2 デフォルト設定
_DEFAULT_TTS_V2 = {
    "checkpoint_v2":          "Aratako/Irodori-TTS-500M-v2",
    "model_device":           "cuda",
    "model_precision":        "fp32",
    "codec_device":           "cuda",
    "codec_precision":        "fp32",
    "num_steps":              20,
    "num_candidates":         1,
    "seed_raw":               "",
    "cfg_guidance_mode":      "independent",
    "cfg_scale_text":         3,
    "cfg_scale_speaker":      5,
    "cfg_scale_raw":          "",
    "context_kv_cache":       True,
    "speaker_kv_scale_raw":   "",
    "speaker_kv_min_t_raw":   "0.9",
    "speaker_kv_max_layers_raw": "",
    "cfg_min_t":              0.5,
    "cfg_max_t":              1,
    "truncation_factor_raw":  "",
    "rescale_k_raw":          "",
    "rescale_sigma_raw":      "",
}

# v3 デフォルト設定
_DEFAULT_TTS_V3 = {
    "checkpoint_v3":          "Aratako/Irodori-TTS-500M-v3",
    "model_device":           "cuda",
    "model_precision":        "fp32",
    "codec_device":           "cuda",
    "codec_precision":        "fp32",
    "num_steps":              20,
    "num_candidates":         1,
    "seed_raw":               "",
    "cfg_guidance_mode":      "independent",
    "cfg_scale_text":         3,
    "cfg_scale_speaker":      5,
    "cfg_scale_raw":          "",
    "context_kv_cache":       True,
    "speaker_kv_scale_raw":   "",
    "speaker_kv_min_t_raw":   "0.9",
    "speaker_kv_max_layers_raw": "",
    "t_schedule_mode":        "sway",
    "sway_coeff":             0.0,
}

_DEFAULT_TTS_V4 = {
    # Empty checkpoint preserves the server default. Blank steps can opt into model defaults.
    "checkpoint_v4": "",
    "num_steps": 20,
    "num_candidates": 1,
    "seed_raw": "",
    "seconds_raw": "",
    "duration_scale": 1.0,
}

DEFAULT_TTS_CONFIG = {"v2": _DEFAULT_TTS_V2, "v3": _DEFAULT_TTS_V3, "v4": _DEFAULT_TTS_V4}

def load_tts_config() -> dict:
    """tts_config.json を読み込む。なければデフォルトを書き出して返す。
    旧形式（平坦な辞書）は自動マイグレーションしてバージョン別構造に変換する。"""
    if os.path.exists(TTS_CONFIG_FILE):
        try:
            with open(TTS_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 旧形式マイグレーション: トップレベルに "v2"/"v3" キーがなければ旧形式
            if not any(version in saved for version in DEFAULT_TTS_CONFIG):
                print("[TTS Config] 旧形式を検出: バージョン別構造にマイグレーションします")
                old_flat = saved
                migrated = {
                    "v2": {**_DEFAULT_TTS_V2,
                           **{k: v for k, v in old_flat.items() if k in _DEFAULT_TTS_V2}},
                    "v3": {**_DEFAULT_TTS_V3,
                           **{k: v for k, v in old_flat.items() if k in _DEFAULT_TTS_V3}},
                    "v4": dict(_DEFAULT_TTS_V4),
                }
                if "checkpoint_v2" in old_flat:
                    migrated["v2"]["checkpoint_v2"] = old_flat["checkpoint_v2"]
                if "checkpoint_v3" in old_flat:
                    migrated["v3"]["checkpoint_v3"] = old_flat["checkpoint_v3"]
                save_tts_config(migrated)
                return migrated
            # 新形式: 各バージョンをデフォルトで補完
            return {version: {**defaults, **saved.get(version, {})}
                    for version, defaults in DEFAULT_TTS_CONFIG.items()}
        except Exception as e:
            print(f"[TTS Config] Failed to load {TTS_CONFIG_FILE}: {e}")
    save_tts_config(DEFAULT_TTS_CONFIG)
    return {version: dict(defaults) for version, defaults in DEFAULT_TTS_CONFIG.items()}

def save_tts_config(data: dict):
    with open(TTS_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[TTS Config] Saved to {TTS_CONFIG_FILE}")

# 起動時に読み込み（グローバル変数として保持）
TTS_CONFIG: dict = load_tts_config()

# バージョン別に /_run_generation へ送るパラメータキーセット
_TTS_V2_SEND_KEYS = frozenset({
    "model_device", "model_precision", "codec_device", "codec_precision",
    "num_steps", "num_candidates", "seed_raw",
    "cfg_guidance_mode", "cfg_scale_text", "cfg_scale_speaker", "cfg_scale_raw",
    "cfg_min_t", "cfg_max_t",
    "context_kv_cache",
    "truncation_factor_raw", "rescale_k_raw", "rescale_sigma_raw",
    "speaker_kv_scale_raw", "speaker_kv_min_t_raw", "speaker_kv_max_layers_raw",
})
_TTS_V3_SEND_KEYS = frozenset({
    "model_device", "model_precision", "codec_device", "codec_precision",
    "num_steps", "num_candidates", "seed_raw",
    "cfg_guidance_mode", "cfg_scale_text", "cfg_scale_speaker", "cfg_scale_raw",
    "t_schedule_mode", "sway_coeff",
    "context_kv_cache",
    "speaker_kv_scale_raw", "speaker_kv_min_t_raw", "speaker_kv_max_layers_raw",
})
_TTS_V4_SEND_KEYS = _TTS_V2_SEND_KEYS | _TTS_V3_SEND_KEYS | {
    "seconds_raw", "duration_scale", "lora_adapter_raw",
}


# --- OpenAI クライアント ---
client = OpenAI(base_url=LLM_API_URL, api_key=LLM_API_KEY)


# 音声管理辞書
# { "req_id": {"audios": [b64,...], "queued": N, "completed": N} }
# queued == completed になったとき「全音声が届いた」と判定できる
finished_audios: dict = {}
finished_audios_lock = threading.Lock()

# --- TTS キュー ---
generate_queue: queue.Queue = queue.Queue()
error_queue = queue.Queue()
tts_generate_lock = threading.Lock()  # tts_client はスレッドセーフでないため排他制御


# --- 共通クライアントの保持 ---
tts_client = None
tts_api_version: str = "v2"      # get_tts_client() 接続時に設定・検出
tts_supports_embedding: bool = False  # /_run_generation に uploaded_speaker_embedding があるか


def detect_tts_version(client) -> str:
    """Gradio APIの互換プロファイルを検出する。ロード中のモデル世代とは別。"""
    try:
        named = client._info.get("named_endpoints", {})
        run_gen = named.get("/_run_generation", {})
        params_list = run_gen.get("parameters", []) if isinstance(run_gen, dict) else []
        param_names = [p.get("parameter_name") or p.get("label", "") for p in params_list]
        version = "v4" if _tts_reference_is_multiple(client) else "v3" if "t_schedule_mode" in param_names else "v2"
        print(f"[TTS] API version detected: {version}")
        return version
    except Exception as e:
        print(f"[TTS] Version detection failed, falling back to v2: {e}")
        return "v2"


def detect_speaker_embedding_support(client) -> bool:
    """接続中のTTSサーバーがSpeaker Embeddingに対応しているか検出する。
    /_run_generation に uploaded_speaker_embedding パラメータがあれば対応。"""
    try:
        named = client._info.get("named_endpoints", {})
        run_gen = named.get("/_run_generation", {})
        params_list = run_gen.get("parameters", []) if isinstance(run_gen, dict) else []
        param_names = [p.get("parameter_name") or p.get("label", "") for p in params_list]
        supported = "uploaded_speaker_embedding" in param_names
        print(f"[TTS] Speaker Embedding support: {supported}")
        return supported
    except Exception as e:
        print(f"[TTS] Speaker Embedding detection failed: {e}")
        return False


def _refresh_embedding_support() -> None:
    """キャラクター選択の都度 TTSサーバーに直接問い合わせて
    tts_supports_embedding を更新する。
    Gradio クライアントを再初期化せずに済む軽量な HTTP リクエスト。
    サーバーが応答しない場合は既存の値を維持する。"""
    global tts_supports_embedding
    try:
        url = TTS_API_URL.rstrip('/') + '/gradio_api/info'
        req = _urllib_req.Request(url)
        with _urllib_req.urlopen(req, timeout=3) as resp:
            info = json.loads(resp.read())
        run_gen    = info.get("named_endpoints", {}).get("/_run_generation", {})
        params     = run_gen.get("parameters", []) if isinstance(run_gen, dict) else []
        names      = [p.get("parameter_name") or p.get("label", "") for p in params]
        tts_supports_embedding = "uploaded_speaker_embedding" in names
        print(f"[TTS] Speaker Embedding support (refreshed): {tts_supports_embedding}")
    except Exception as e:
        print(f"[TTS] Embedding support refresh failed, keeping current value ({tts_supports_embedding}): {e}")


def get_tts_client():
    """Gradio Clientをシングルトンで取得。接続時にAPIバージョンとEmbedding対応を検出する。"""
    global tts_client, tts_api_version, tts_supports_embedding
    if tts_client is None:
        try:
            print(f"Connecting to TTS Server at {TTS_API_URL}...")
            tts_client = Client(TTS_API_URL)
            configured = load_settings().get("tts_api_version", "auto")
            if configured in DEFAULT_TTS_CONFIG:
                tts_api_version = configured
                print(f"[TTS] API version (manual): {tts_api_version}")
            else:
                tts_api_version = detect_tts_version(tts_client)
            tts_supports_embedding = detect_speaker_embedding_support(tts_client)
            print(f"Successfully connected to TTS Server (API: {tts_api_version}).")
        except Exception as e:
            print(f"Failed to connect to TTS Server: {e}")
    return tts_client


# --- TTS 生成ワーカー ---

def tts_generator_worker():
    """裏側で音声を生成し続けるワーカー"""
    while True:
        item = generate_queue.get()
        if item is None:
            break

        text, voice_path, embedding_path, req_id = item
        try:
            audio_b64 = generate_and_encode_tts(text, voice_path, embedding_path)
        except Exception as e:
            print(f"Worker Error: {e}")
            audio_b64 = None
        finally:
            # 成否に関わらず completed をインクリメントし、音声があれば追加
            with finished_audios_lock:
                if req_id in finished_audios:
                    if audio_b64:
                        finished_audios[req_id]["audios"].append(audio_b64)
                        print(f"[TTS] Generated for {req_id}: {text[:20]}...")
                    else:
                        print(f"[TTS] Failed (skipped) for {req_id}: {text[:20]}...")
                    finished_audios[req_id]["completed"] += 1
            generate_queue.task_done()

# スレッドを開始
threading.Thread(target=tts_generator_worker, daemon=True).start()


# --- キャラクター読み込み ---

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
# zip展開キャッシュ先
ZIP_CACHE_DIR = os.path.join("characters", ".zip_cache")


def _find_image(directory: str, base: str) -> str | None:
    """directory 内で base 名に一致する画像ファイルパスを返す"""
    for ext in IMAGE_EXTS:
        p = os.path.join(directory, base + ext)
        if os.path.exists(p):
            return p
    return None


def _extract_zip(zip_path: str) -> str:
    """
    zip を ZIP_CACHE_DIR/<stem>/ に展開し、展開先ディレクトリを返す。
    既に展開済みかつ zip より新しければ再展開しない。
    """
    os.makedirs(ZIP_CACHE_DIR, exist_ok=True)
    stem = Path(zip_path).stem
    dest = os.path.join(ZIP_CACHE_DIR, stem)

    zip_mtime  = os.path.getmtime(zip_path)
    dest_mtime = os.path.getmtime(dest) if os.path.exists(dest) else 0

    if dest_mtime < zip_mtime:
        # 再展開
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(dest, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Windows(7-Zip等)でCP932エンコードされたファイル名に対応
                # UTF-8フラグが立っていないエントリはCP932として再デコードする
                for info in zf.infolist():
                    if not (info.flag_bits & 0x800):  # UTF-8フラグなし
                        try:
                            info.filename = info.filename.encode('cp437').decode('cp932')
                        except (UnicodeDecodeError, UnicodeEncodeError):
                            pass  # デコード失敗時はそのまま使用
                    zf.extract(info, dest)
            print(f"[ZIP] Extracted: {zip_path} → {dest}")
        except Exception as e:
            print(f"[ZIP] Failed to extract {zip_path}: {e}")
            shutil.rmtree(dest, ignore_errors=True)
            return ""
    return dest


def _char_from_dir(base: str, directory: str) -> dict | None:
    """
    directory 内に <base>.txt があればキャラクター辞書を返す。
    txt が存在しない場合は None。
    音声は .wav を優先し、なければ .mp3 を使用する。
    """
    txt_path = os.path.join(directory, base + ".txt")
    if not os.path.exists(txt_path):
        # サブフォルダ内でフォルダ名と異なるtxtを探す
        txts = [f for f in os.listdir(directory) if f.endswith(".txt")]
        if not txts:
            return None
        txt_path = os.path.join(directory, txts[0])
        base = Path(txts[0]).stem

    wav_path = os.path.join(directory, base + ".wav")
    mp3_path = os.path.join(directory, base + ".mp3")
    has_wav  = os.path.exists(wav_path)
    has_mp3  = os.path.exists(mp3_path)

    if has_wav:
        active_voice = wav_path
    elif has_mp3:
        active_voice = mp3_path
    else:
        # 同名でなくとも .wav / .mp3 が1つだけあれば使う（.wav 優先）
        wavs = [f for f in os.listdir(directory) if f.endswith(".wav")]
        mp3s = [f for f in os.listdir(directory) if f.endswith(".mp3")]
        if wavs:
            wav_path     = os.path.join(directory, wavs[0])
            active_voice = wav_path
            has_wav      = True
            if mp3s:
                mp3_path = os.path.join(directory, mp3s[0])
                has_mp3  = True
        elif mp3s:
            mp3_path     = os.path.join(directory, mp3s[0])
            active_voice = mp3_path
            has_mp3      = True
        else:
            active_voice = None

    has_voice = active_voice is not None

    # Speaker Embedding: <base>.speaker.safetensors のみを認識（完全一致）
    # フラット配置では複数キャラが同一ディレクトリを共有するため fallback は使わない
    embedding_path = None
    try:
        target = base + '.speaker.safetensors'
        candidate = os.path.join(directory, target)
        if not os.path.exists(candidate):
            # 大文字小文字が異なるケースに対応（Windows以外）
            target_lower = target.lower()
            candidate = next(
                (os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower() == target_lower),
                None,
            )
        else:
            candidate = candidate
        embedding_path = candidate if candidate and os.path.exists(candidate) else None
    except OSError:
        pass

    return {
        "name":           base if has_voice else f"{base} (音声なし)",
        "prompt":         txt_path,
        "voice_path":     active_voice,
        "mp3_path":       mp3_path if has_mp3 else None,
        "image_path":     _find_image(directory, base),
        "has_voice":      has_voice,
        "has_mp3":        has_mp3,
        "embedding_path": embedding_path,
        "has_embedding":  embedding_path is not None,
    }


def _scan_char_dir(scan_dir: str, char_list: list, seen_names: set) -> None:
    """scan_dir 内のキャラクターを char_list に追加する。seen_names に既にある名前はスキップ。"""
    if not os.path.exists(scan_dir):
        return

    for entry in sorted(os.listdir(scan_dir)):
        full_path = os.path.join(scan_dir, entry)

        # ── パターン1: フラット (.txt) ──────────────────────────
        if entry.endswith(".txt"):
            base = Path(entry).stem
            if base in seen_names:
                continue
            char = _char_from_dir(base, scan_dir)
            if char:
                char["char_format"] = "flat"
                seen_names.add(base)
                char_list.append(char)

        # ── パターン2: サブフォルダ ──────────────────────────────
        elif os.path.isdir(full_path) and entry != ".zip_cache":
            base = entry
            if base in seen_names:
                continue
            char = _char_from_dir(base, full_path)
            if char:
                char["char_format"] = "subfolder"
                seen_names.add(base)
                char_list.append(char)

        # ── パターン3: zip ───────────────────────────────────────
        elif entry.endswith(".zip"):
            base = Path(entry).stem
            if base in seen_names:
                continue
            dest = _extract_zip(full_path)
            if not dest:
                continue
            # 展開先がフォルダ1つだけの場合はその中を見る
            sub_entries = os.listdir(dest)
            if len(sub_entries) == 1 and os.path.isdir(os.path.join(dest, sub_entries[0])):
                search_dir = os.path.join(dest, sub_entries[0])
            else:
                search_dir = dest
            char = _char_from_dir(base, search_dir)
            if not char:
                # stem名と一致しなくてもtxtを探す
                char = _char_from_dir(
                    Path([f for f in os.listdir(search_dir) if f.endswith(".txt")][0]).stem
                    if any(f.endswith(".txt") for f in os.listdir(search_dir)) else base,
                    search_dir,
                )
            if char:
                char["char_format"] = "zip"
                seen_names.add(base)
                char_list.append(char)


def get_dynamic_characters() -> list:
    # CHAR_DIR が存在しない場合は作成して空を返す
    if not os.path.exists(CHAR_DIR):
        os.makedirs(CHAR_DIR)
        return []

    char_list: list[dict] = []
    seen_names: set[str]  = set()   # 重複防止 (CHAR_DIR 側が優先)

    # CHAR_DIR を先にスキャン (優先度高)
    _scan_char_dir(CHAR_DIR, char_list, seen_names)

    # CHAR_DIR 環境変数が有効 かつ char_dir_also_default=True の場合は
    # デフォルトフォルダも追加でスキャン (同名はCHAR_DIR側を優先済み)
    settings = load_settings()
    if CHAR_DIR_ENV and settings.get("char_dir_also_default", False):
        if os.path.normpath(CHAR_DIR) != os.path.normpath(DEFAULT_CHAR_DIR):
            _scan_char_dir(DEFAULT_CHAR_DIR, char_list, seen_names)

    return char_list


# --- アプリ状態 ---
class AppState:
    def __init__(self):
        self.chars:          list  = []
        self.selected:       dict  = {}
        self.memory:         list  = []
        self.base_prompt:    str   = ""
        self.mem_pending:    list  = []    # 記憶抽出待ちの発言 (人間の入力とその応答のみ)
        self.mem_user_turns: int   = 0     # mem_pending 内のユーザー発言数
        self.session_ctx:    dict  = {}    # 会話開始時の状況 (前回からの経過など)

state = AppState()

# Normal chat and game mode share only this short reservation gate. The lock is
# never held across an LLM request or a streaming response.
_conversation_gate = threading.RLock()
_conversation_epoch = str(uuid.uuid4())
_normal_revision = 0
_normal_generations: dict[str, dict] = {}
_view_reset_records: dict[tuple[str, str], tuple[str, dict, float]] = {}
_game_transcript_writer = GameTranscriptWriter()


def _conversation_snapshot() -> tuple[str, int]:
    return _conversation_epoch, _normal_revision


class _GameLLMFactory:
    def snapshot(self):
        settings = load_settings()
        client = OpenAI(
            base_url=_build_llm_url(settings),
            api_key=settings.get("llm_api_key") or "lm-studio",
        )
        return OpenAIGameLLM(client, settings.get("llm_model") or "local-model")


def _save_game_transcript(session) -> str | None:
    if not load_settings().get("log_auto_save", True):
        return None
    try:
        _game_transcript_writer.save(session)
        return None
    except Exception:
        return "公開ログを自動保存できませんでした。記録はこのゲーム中のRAMに保持しています。"


def _normal_busy() -> bool:
    if _normal_generations:
        return True
    audio_lock = globals().get("finished_audios_lock")
    audio_state = globals().get("finished_audios")
    if audio_lock is not None and audio_state is not None:
        with audio_lock:
            return any(item.get("queued", 0) > item.get("completed", 0) for item in audio_state.values())
    return False


_game_engine = GameEngine(
    _GameLLMFactory(),
    gate_lock=_conversation_gate,
    normal_busy=_normal_busy,
    conversation_state=_conversation_snapshot,
    transcript_save=_save_game_transcript,
)


def _reserve_normal_generation(request: Request, *, bump_revision: bool = True) -> str:
    global _normal_revision
    page_id = request.headers.get("X-Bridge-Page-Id")
    epoch = request.headers.get("X-Conversation-Epoch")
    with _conversation_gate:
        if epoch and epoch != _conversation_epoch:
            raise HTTPException(409, detail={"code": "STALE_CONVERSATION"})
        try:
            _game_engine.guard_normal(page_id, epoch)
        except GameError as exc:
            raise HTTPException(exc.status_code, detail={"code": exc.code}) from exc
        token = str(uuid.uuid4())
        _normal_generations[token] = {"page_id": page_id, "epoch": _conversation_epoch}
        if bump_revision:
            _normal_revision += 1
        return token


def _release_normal_generation(token: str):
    with _conversation_gate:
        _normal_generations.pop(token, None)


def _tracked_normal_stream(stream, token: str):
    try:
        yield from stream
    finally:
        _release_normal_generation(token)


def _conversation_transition(request: Request | None = None, *, page_id: str | None = None, epoch: str | None = None, view_reset: bool = False):
    global _conversation_epoch, _normal_revision
    page_id = page_id or (request.headers.get("X-Bridge-Page-Id") if request else None)
    epoch = epoch or (request.headers.get("X-Conversation-Epoch") if request else None)
    with _conversation_gate:
        if _normal_generations:
            raise HTTPException(409, detail={"code": "GAME_BUSY"})
        if epoch and epoch != _conversation_epoch:
            raise HTTPException(409, detail={"code": "STALE_CONVERSATION"})
        if view_reset:
            session = _game_engine.current(page_id or "")
            if session and session.status != "suspended":
                raise HTTPException(409, detail={"code": "GAME_BUSY"})
        try:
            _game_engine.transition(page_id, epoch)
        except GameError as exc:
            raise HTTPException(exc.status_code, detail={"code": exc.code}) from exc
        _conversation_epoch = str(uuid.uuid4())
        _normal_revision = 0
        return {"conversation_epoch": _conversation_epoch, "normal_revision": _normal_revision}

# --- FastAPI ---
app = FastAPI(title="Bridge WebUI")


@app.exception_handler(GameError)
async def _game_error_handler(request: Request, exc: GameError):
    return JSONResponse(status_code=exc.status_code, content={"detail": {"code": exc.code}})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自動起動用
def open_browser():
    port = os.getenv("BRIDGE_PORT", "8001")
    url = f"http://localhost:{port}"
    time.sleep(2)
    print(f"[INFO] Opening browser at {url}")
    webbrowser.open(url)


# --- index.html 配信 ---
@app.get("/")
def root():
    return FileResponse("index.html")

# --- キャラクター一覧 ---
@app.get("/characters")
def list_characters():
    state.chars = get_dynamic_characters()
    return [{
        "index":         i,
        "name":          c["name"],
        "has_image":     c["image_path"] is not None,
        "has_voice":     c["has_voice"],
        "has_embedding": c.get("has_embedding", False),
    } for i, c in enumerate(state.chars)]

# --- キャラクター画像 ---
@app.get("/characters/{idx}/image")
def get_character_image(idx: int):
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404)

    img_path = state.chars[idx].get("image_path")
    if not img_path or not os.path.exists(img_path):
        raise HTTPException(404)

    with Image.open(img_path) as source:
        # Palette PNGs can carry transparency without an alpha band.
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        img = source.convert("RGBA" if has_alpha else "RGB")
    img = crop_center(img)
    img = img.resize((640, 640), Image.LANCZOS)

    buf = io.BytesIO()
    if has_alpha:
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png" if has_alpha else "image/jpeg")

def crop_center(img):
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = 0
    return img.crop((left, top, left + size, top + size))

# --- システムプロンプト合成ヘルパー ---

def _read_file_prompt(fname: str) -> str:
    """system_prompt_file の値からテキストを読んで返す。"""
    if not fname:
        return ""
    fpath = os.path.join(PROMPTS_DIR, fname)
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def _get_active_sys_prompt_text(settings: dict) -> str:
    """設定からアクティブなシステムプロンプトテキストを返す。

    slot_and_file=True のとき: ファイル + スロットを両方返す（改行2つで連結）
    slot_and_file=False のとき: スロットが有効ならスロットのみ、なければファイル
    """
    slot          = settings.get("active_user_slot", -1)
    slots         = settings.get("user_slots", ["", "", "", "", ""])
    fname         = settings.get("system_prompt_file", "")
    slot_and_file = settings.get("slot_and_file", False)

    slot_text = ""
    if isinstance(slot, int) and 0 <= slot < len(slots):
        slot_text = (slots[slot] or "").strip()

    file_text = _read_file_prompt(fname)

    if slot_and_file:
        # 同時利用: ファイル → スロットの順で連結
        parts = [t for t in [file_text, slot_text] if t]
        return "\n\n".join(parts)
    else:
        # 従来動作: スロットが有効ならスロットだけ、なければファイル
        if slot_text:
            return slot_text
        return file_text


def _build_system_prompt(base_prompt: str, char_setting: str, settings: dict) -> str:
    """sys_prompt + char_setting を設定順で合成する。
    base_prompt 引数は後方互換のため残すが使用しない
    sys_prompt と char_setting を設定順で合成する。"""
    sys_text   = _get_active_sys_prompt_text(settings)
    after_char = settings.get("system_prompt_after_char", False)

    parts = []
    if after_char:
        if char_setting:
            parts.append(char_setting)
        if sys_text:
            parts.append(sys_text)
    else:
        if sys_text:
            parts.append(sys_text)
        if char_setting:
            parts.append(char_setting)

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════
#  キャラクター長期記憶 / 関係性 (memory/<キャラ名>.json)
# ══════════════════════════════════════════════════════

from datetime import datetime as _datetime

MEMORY_DIR          = "memory"
MEMORY_MAX_FACTS    = 40   # ユーザーに関する事実の上限
MEMORY_MAX_EPISODES = 20   # エピソード要約の上限
MEMORY_MAX_PROMISES = 10   # 約束の上限
SESSION_GAP_MINUTES = 30   # この分数以上空いたら新しい会話(セッション)とみなす

_memory_file_lock   = threading.Lock()   # ファイル読み書きの排他
_memory_update_lock = threading.Lock()   # LLM抽出処理の直列化
_memory_updating: set = set()            # 抽出中のキャラキー

# (親密度の下限, 段階名, 振る舞いのヒント)
RELATIONSHIP_STAGES = [
    (0,  "まだ手探り",     "相手のことをまだよく知らない。少し距離を保ち、様子をうかがいながら話す。踏み込んだ話や甘えは控えめ。"),
    (20, "打ち解けてきた", "気軽に話せるようになってきた。冗談や本音が少しずつ出る。相手の話を覚えていて時々持ち出す。"),
    (45, "気心の知れた仲", "遠慮が減り、自分の話や弱みも見せる。相手の癖や好みを踏まえてからかったり気遣ったりする。"),
    (70, "特別な相手",     "強い信頼と愛着がある。素の自分で接し、相手の機嫌や小さな変化に敏感に反応する。沈黙も気にしない。"),
]


def _memory_key(char_name: str) -> str:
    """キャラ名からメモリファイル名に使う安全なキーを作る"""
    name = (char_name or "").replace(" (音声なし)", "").strip()
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    return name or "unknown"


def _memory_path(char_name: str) -> str:
    return os.path.join(MEMORY_DIR, _memory_key(char_name) + ".json")


def _default_memory(char_name: str) -> dict:
    return {
        "char_name":            _memory_key(char_name),
        "relationship_enabled": False,  # キャラ設定: 関係性の変化を有効にする (既定: 無効)
        "user_facts":           [],     # ユーザーに関する事実
        "episodes":             [],     # {"date": "YYYY-MM-DD", "summary": "..."}
        "promises":             [],     # 約束・次に話すこと
        "affinity":             10,     # 親密度 0〜100
        "relationship_note":    "",     # キャラ視点のユーザーへの印象
        "sessions":             0,      # 会話(セッション)回数
        "turns":                0,      # 記憶対象になったユーザー発言の累計
        "last_seen":            None,   # 最後に会話した日時 (ISO)
    }


def load_memory(char_name: str) -> dict:
    path = _memory_path(char_name)
    mem  = _default_memory(char_name)
    with _memory_file_lock:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    mem.update({k: v for k, v in saved.items() if k in mem})
            except Exception as e:
                print(f"[Memory] Failed to load {path}: {e}")
    return mem


def save_memory(char_name: str, mem: dict) -> None:
    """アトミック書き込み (save_settings と同方式)"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    path = _memory_path(char_name)
    with _memory_file_lock:
        fd, tmp_path = tempfile.mkstemp(dir=MEMORY_DIR, suffix=".tmp", prefix="mem_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(mem, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise


def reset_memory(char_name: str) -> dict:
    """記憶を初期状態に戻す。キャラ設定(relationship_enabled)は保持する"""
    old = load_memory(char_name)
    mem = _default_memory(char_name)
    mem["relationship_enabled"] = bool(old.get("relationship_enabled", False))
    save_memory(char_name, mem)
    return mem


def set_relationship_enabled(char_name: str, enabled: bool) -> None:
    mem = load_memory(char_name)
    mem["relationship_enabled"] = bool(enabled)
    save_memory(char_name, mem)


def relationship_stage(affinity: int) -> tuple[str, str]:
    """親密度から (段階名, 振る舞いヒント) を返す"""
    stage = RELATIONSHIP_STAGES[0]
    for s in RELATIONSHIP_STAGES:
        if affinity >= s[0]:
            stage = s
    return stage[1], stage[2]


def _time_of_day(now: _datetime) -> str:
    h = now.hour
    if 4 <= h < 7:   return "早朝"
    if 7 <= h < 11:  return "朝"
    if 11 <= h < 14: return "昼"
    if 14 <= h < 17: return "午後"
    if 17 <= h < 19: return "夕方"
    if 19 <= h < 23: return "夜"
    return "深夜"


def _format_now(now: _datetime) -> str:
    wd = "月火水木金土日"[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日({wd}) {now.hour}時{now.minute:02d}分 ({_time_of_day(now)})"


def _elapsed_text(last_seen_iso, now: _datetime) -> str:
    """前回会話からの経過を自然な日本語で返す"""
    if not last_seen_iso:
        return "初めての会話"
    try:
        last = _datetime.fromisoformat(last_seen_iso)
    except Exception:
        return ""
    delta = now - last
    mins  = int(delta.total_seconds() // 60)
    if mins < SESSION_GAP_MINUTES: return "さっきまで話していた続き"
    if mins < 180:                 return f"{mins // 60 or 1}時間ほど前にも話した"
    if delta.days < 1:             return "今日すでに一度話している"
    if delta.days == 1:            return "昨日ぶり"
    if delta.days < 7:             return f"{delta.days}日ぶり"
    if delta.days < 30:            return f"{delta.days // 7}週間ぶり"
    if delta.days < 365:           return f"{delta.days // 30}か月ぶり"
    return "1年以上ぶり"


def _build_context_block(settings: dict) -> str:
    """system プロンプト末尾に付ける「現在の状況 / 記憶 / 関係性」ブロックを組み立てる"""
    now   = _datetime.now()
    lines = ["# 現在の状況", f"- 現在の日時: {_format_now(now)}"]

    if not settings.get("memory_enabled", True) or not state.selected:
        return "\n".join(lines)

    mem     = load_memory(state.selected["name"])
    sctx    = state.session_ctx or {}
    elapsed = sctx.get("elapsed") or _elapsed_text(mem.get("last_seen"), now)
    sess    = int(mem.get("sessions", 0))
    if elapsed == "初めての会話" or (sess <= 1 and int(mem.get("turns", 0)) == 0):
        lines.append("- ユーザーとの会話: これが初めての会話。")
    else:
        lines.append(f"- ユーザーとの会話: {sess}回目。{elapsed}。")

    facts    = [f for f in mem.get("user_facts", []) if f]
    episodes = mem.get("episodes", [])[-6:]
    promises = [p for p in mem.get("promises", []) if p]

    if facts or episodes or promises:
        lines.append("")
        lines.append("# ユーザーについての記憶(会話の中で自然に活かす。列挙したり「覚えてるよ」と確認したりしない)")
        for f in facts:
            lines.append(f"- {f}")
        if episodes:
            lines.append("## 最近の会話で起きたこと")
            for ep in episodes:
                lines.append(f"- {ep.get('date', '')}: {ep.get('summary', '')}".rstrip())
        if promises:
            lines.append("## 約束・次に話そうと言っていたこと")
            for p in promises:
                lines.append(f"- {p}")

    if mem.get("relationship_enabled"):
        affinity    = int(mem.get("affinity", 10))
        stage, hint = relationship_stage(affinity)
        lines.append("")
        lines.append("# 関係性(【ペルソナ設定】の関係を土台に、今の距離感として反映する)")
        lines.append(f"- 今の段階: {stage} (親密度 {affinity}/100)")
        lines.append(f"- 距離感の目安: {hint}")
        if mem.get("relationship_note"):
            lines.append(f"- あなたが相手に抱いている印象: {mem['relationship_note']}")
    return "\n".join(lines)


def _begin_session(char_name: str, settings: dict, private: bool = False) -> None:
    """キャラ選択時に呼ぶ。前回からの経過を記録し、必要ならセッション数を増やす。
    private=True (プライベートモード) のときはファイルを更新しない"""
    state.session_ctx = {}
    if not settings.get("memory_enabled", True):
        return
    now     = _datetime.now()
    mem     = load_memory(char_name)
    elapsed = _elapsed_text(mem.get("last_seen"), now)
    new_session = True
    if mem.get("last_seen"):
        try:
            gap = now - _datetime.fromisoformat(mem["last_seen"])
            new_session = gap.total_seconds() >= SESSION_GAP_MINUTES * 60
        except Exception:
            pass
    if not private:
        if new_session:
            mem["sessions"] = int(mem.get("sessions", 0)) + 1
        mem["last_seen"] = now.isoformat(timespec="seconds")
        save_memory(char_name, mem)
    state.session_ctx = {"elapsed": elapsed, "new_session": new_session}


def _touch_memory_turn(char_name: str) -> None:
    """記憶対象の1往復が完了したときに呼ぶ (最終会話日時・発言数を更新)"""
    try:
        mem = load_memory(char_name)
        mem["turns"]     = int(mem.get("turns", 0)) + 1
        mem["last_seen"] = _datetime.now().isoformat(timespec="seconds")
        save_memory(char_name, mem)
    except Exception as e:
        print(f"[Memory] touch failed: {e}")


def _greeting_instruction(settings: dict) -> str:
    """会話開始時の挨拶指示。時間帯と前回からの経過に合わせて変える"""
    now  = _datetime.now()
    base = (f"会話開始時の最初の一言をください。今は{_time_of_day(now)}です。"
            "挨拶だけで終わらず、今の自分の様子や気分、ふと思ったことをひとこと添えて、"
            "自然に話しかけてください。1〜2文。")
    if not settings.get("memory_enabled", True):
        return base
    elapsed = (state.session_ctx or {}).get("elapsed", "")
    if elapsed == "初めての会話":
        return base + "相手とはまだ知り合ったばかりなので、距離感に気をつけて。"
    if elapsed == "さっきまで話していた続き":
        return base + "さっきまで話していた続きなので、改めての挨拶は不要です。"
    if elapsed:
        return base + f"相手と話すのは{elapsed}です。それを踏まえた一言にしてください。記憶にあることに触れてもかまいません。"
    return base


# ── LLM による記憶抽出 ──────────────────────────────────────────────

_MEMORY_EXTRACT_PROMPT = """あなたは会話ログを読み、キャラクターが相手(ユーザー)について覚えておくべきことを整理する係です。
キャラクター自身の発言や設定は記憶の対象ではありません。ユーザーについて分かったことだけを扱います。

次のJSONオブジェクトだけを出力してください。説明文やコードブロックは不要です。
{
  "new_user_facts": ["ユーザー本人に関する新しい事実(名前・呼び方・好き嫌い・仕事や学校・家族・住んでいる場所・近況など)。既知の事実と重複するものは含めない。なければ空配列"],
  "episode_summary": "今回の会話で起きたことの要約(1〜2文、日本語)。特筆すべきことがなければ空文字",
  "promises": ["約束したこと・次に話そうと言ったこと。なければ空配列"],
  "resolved_promises": ["既存の約束のうち果たされた・不要になったもの(既存の文言をそのまま)。なければ空配列"],
  "affinity_delta": 0,
  "relationship_note": "キャラクターの視点で見た、ユーザーへの今の印象(1〜2文)。前回の印象を踏まえて更新する"
}

affinity_delta は -5〜5 の整数。楽しい・心が通った・相手が自分を気遣ってくれた会話は +1〜+5、
冷たい・そっけない・不快な会話は -1〜-5、普通の雑談は 0〜+1。"""


def _extract_json(text: str) -> dict | None:
    """LLM出力からJSONオブジェクトを取り出す (コードブロック・前後の説明文に耐える)"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_str_list(v) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


def _run_memory_update(char_name: str, transcript: list[dict]) -> None:
    """バックグラウンドスレッドで実行: transcript から記憶を抽出して保存する"""
    key = _memory_key(char_name)
    with _memory_update_lock:
        _memory_updating.add(key)
        try:
            mem         = load_memory(char_name)
            known_facts = list(mem.get("user_facts", []))
            convo = "\n".join(
                f"{'ユーザー' if m['role'] == 'user' else 'キャラクター'}: {m['content']}"
                for m in transcript if m.get("content", "").strip()
            )
            user_msg = (
                f"【キャラクター名】{key}\n"
                "【既知のユーザー情報】\n" + ("\n".join(f"- {f}" for f in known_facts) or "- (なし)") + "\n"
                "【既存の約束】\n" + ("\n".join(f"- {p}" for p in mem.get("promises", [])) or "- (なし)") + "\n"
                f"【前回までの印象】{mem.get('relationship_note') or '(なし)'}\n\n"
                f"【会話ログ】\n{convo}"
            )
            resp = client.chat.completions.create(
                model=LLM_MODEL or "local-model",
                messages=[
                    {"role": "system", "content": _MEMORY_EXTRACT_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                stream=False,
            )
            raw  = (resp.choices[0].message.content or "").strip()
            data = _extract_json(raw)
            if data is None:
                print(f"[Memory] JSON parse failed for {key}: {raw[:120]!r}")
                return

            # 事実 (重複除去・上限)
            new_facts = _as_str_list(data.get("new_user_facts"))
            facts = known_facts
            for f in new_facts:
                if f not in facts:
                    facts.append(f)
            mem["user_facts"] = facts[-MEMORY_MAX_FACTS:]

            # エピソード
            summary = str(data.get("episode_summary") or "").strip()
            if summary:
                today = _datetime.now().strftime("%Y-%m-%d")
                mem.setdefault("episodes", []).append({"date": today, "summary": summary})
                mem["episodes"] = mem["episodes"][-MEMORY_MAX_EPISODES:]

            # 約束
            resolved = set(_as_str_list(data.get("resolved_promises")))
            promises = [p for p in mem.get("promises", []) if p not in resolved]
            for p in _as_str_list(data.get("promises")):
                if p not in promises:
                    promises.append(p)
            mem["promises"] = promises[-MEMORY_MAX_PROMISES:]

            # 親密度・印象 (キャラ設定で関係性の変化が有効なときのみ)
            delta = 0
            if mem.get("relationship_enabled"):
                try:
                    delta = int(data.get("affinity_delta", 0))
                except (TypeError, ValueError):
                    delta = 0
                delta = max(-5, min(5, delta))
                mem["affinity"] = max(0, min(100, int(mem.get("affinity", 10)) + delta))
                note = str(data.get("relationship_note") or "").strip()
                if note:
                    mem["relationship_note"] = note[:200]

            save_memory(char_name, mem)
            print(f"[Memory] Updated {key}: +{len(new_facts)} facts, affinity {delta:+d} -> {mem['affinity']}")
        except Exception as e:
            print(f"[Memory] Update failed for {key}: {e}")
        finally:
            _memory_updating.discard(key)


def _schedule_memory_update(char_name: str | None = None) -> bool:
    """mem_pending を取り出してバックグラウンドで記憶更新を開始する。開始したら True"""
    name = char_name or (state.selected.get("name") if state.selected else None)
    transcript = list(state.mem_pending)
    state.mem_pending    = []
    state.mem_user_turns = 0
    if not name or not transcript:
        return False
    threading.Thread(target=_run_memory_update, args=(name, transcript), daemon=True).start()
    return True


def _record_memory_turn(user_text: str, assistant_text: str, settings: dict) -> None:
    """記憶対象の1往復を mem_pending に積み、間隔に達したら抽出を開始する"""
    if not settings.get("memory_enabled", True) or not state.selected:
        return
    if not user_text.strip() or not assistant_text.strip():
        return
    state.mem_pending.append({"role": "user",      "content": user_text})
    state.mem_pending.append({"role": "assistant", "content": assistant_text})
    state.mem_user_turns += 1
    _touch_memory_turn(state.selected["name"])
    try:
        interval = max(1, min(30, int(settings.get("memory_update_interval", 6))))
    except (TypeError, ValueError):
        interval = 6
    if state.mem_user_turns >= interval:
        _schedule_memory_update()


def _build_llm_messages(settings: dict) -> list[dict]:
    """LLMへ渡すメッセージ列を組み立てる。
    system = 保持中のシステムプロンプト + 現在の状況/記憶/関係性ブロック
    履歴  = history_max_messages で直近に絞る (0=無制限)"""
    if not state.memory:
        return []
    has_sys     = state.memory[0].get("role") == "system"
    sys_content = state.memory[0]["content"] if has_sys else ""
    history     = state.memory[1:] if has_sys else list(state.memory)

    ctx = _build_context_block(settings)
    if ctx:
        sys_content = (sys_content + "\n\n" + ctx).strip()

    try:
        max_msgs = int(settings.get("history_max_messages", 30))
    except (TypeError, ValueError):
        max_msgs = 30
    if max_msgs > 0 and len(history) > max_msgs:
        history = history[-max_msgs:]

    return [{"role": "system", "content": sys_content}] + history


# --- キャラクター選択 ---
class SelectRequest(BaseModel):
    index: int
    private: bool = False   # True=プライベートモード (記憶ファイルを更新しない)

@app.post("/select")
def select_character(req: SelectRequest, request: Request):
    state.chars = get_dynamic_characters()
    if req.index < 0 or req.index >= len(state.chars):
        raise HTTPException(400, "Invalid index")

    prev_name = state.selected.get("name") if state.selected else None
    selected = state.chars[req.index]

    base_prompt = ""
    if os.path.exists(BASE_PROMPT_FILE):
        with open(BASE_PROMPT_FILE, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()

    with open(selected["prompt"], "r", encoding="utf-8") as f:
        char_setting = f.read().strip()

    # システムプロンプトの合成
    settings = load_settings()
    sys_prompt_text = _build_system_prompt(base_prompt, char_setting, settings)

    conversation = _conversation_transition(request)
    state.selected = selected

    state.memory = [{"role": "system", "content": sys_prompt_text}]

    # 前のキャラの未反映の会話を記憶に書き出し、新しいキャラのセッションを開始する
    if prev_name:
        _schedule_memory_update(prev_name)
    else:
        state.mem_pending    = []
        state.mem_user_turns = 0
    _begin_session(state.selected["name"], settings, private=req.private)

    # キャラ選択のたびにEmbedding対応を再確認（TTSサーバーが再起動された可能性があるため）
    _refresh_embedding_support()

    # Speaker Embedding非対応APIの場合に警告メッセージを生成
    embedding_warning = None
    if state.selected.get("has_embedding") and not tts_supports_embedding:
        if state.selected.get("has_voice"):
            embedding_warning = "APIがSpeaker Embedding非対応です。リファレンスオーディオを使用します。"
        elif settings.get("no_voice_mode") == "default_voice":
            embedding_warning = "APIがSpeaker Embedding非対応です。標準音声を使用します。"
        else:
            embedding_warning = "APIがSpeaker Embedding非対応です。"

    return {"status": "ok", "name": state.selected["name"], "embedding_warning": embedding_warning, **conversation}


@app.post("/update_system_prompt")
def update_system_prompt(request: Request):
    """システムプロンプトを再構築し、会話履歴をリセットする。
    LLMへ渡す履歴は system メッセージのみになるため文脈の汚染を防ぐ。
    UI側のログ表示はフロントエンドが保持するため見た目は変化しない。"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")

    with open(state.selected["prompt"], "r", encoding="utf-8") as f:
        char_setting = f.read().strip()

    settings = load_settings()
    sys_prompt_text = _build_system_prompt("", char_setting, settings)

    conversation = _conversation_transition(request)

    # 会話履歴をリセット。ただし最後のキャラ発言は保持して
    # 自動応答が直後にトリガーされても /user_reply_gen が機能するようにする。
    last_assistant = next(
        (m for m in reversed(list(state.memory)) if m.get("role") == "assistant"), None
    )
    state.memory = [{"role": "system", "content": sys_prompt_text}]
    if last_assistant:
        state.memory.append(last_assistant)

    return {"status": "ok", **conversation}


@app.get("/status")
async def get_status():
    is_processing = not generate_queue.empty()
    return {"is_generating": is_processing}


# --- チャット ---
class ChatRequest(BaseModel):
    message: str
    tts_enabled: bool = True
    memorize: bool = True   # False=自動応答・シナリオ・プライベートモード (記憶に反映しない)


# --- 補助関数 ---
def _handle_file_with_retry(file_path: str, label: str):
    """handle_file を最大2回試行して結果を返す。失敗時は None。"""
    for attempt in range(2):
        try:
            result_hf = handle_file(file_path)
            print(f"[TTS] handle_file({label}) attempt {attempt + 1}: type={type(result_hf).__name__} value={result_hf!r}")
            if result_hf is not None and result_hf != "" and result_hf != {}:
                return result_hf
            print(f"[TTS] WARNING: handle_file({label}) が空値を返しました (attempt {attempt + 1}): {result_hf!r}")
        except Exception as hf_err:
            print(f"[TTS] handle_file({label}) attempt {attempt + 1} failed: {hf_err}")
    return None


def _tts_reference_is_multiple(client) -> bool:
    """Inspect only the reference input's schema, independently of model generation."""
    info = getattr(client, "_info", {})
    if not isinstance(info, dict):
        return False
    endpoint = info.get("named_endpoints", {}).get("/_run_generation", {})
    parameters = endpoint.get("parameters", []) if isinstance(endpoint, dict) else []
    parameter = next((item for item in parameters if item.get("parameter_name") == "uploaded_audio"), {})
    root = parameter.get("type", {})

    def is_array(schema, depth=0):
        if not isinstance(schema, dict) or depth > 8:
            return False
        kind = schema.get("type")
        if kind == "array" or isinstance(kind, list) and "array" in kind:
            return True
        reference = schema.get("$ref", "")
        if isinstance(reference, str) and reference.startswith("#/"):
            target = root
            for part in reference[2:].split("/"):
                target = target.get(part.replace("~1", "/").replace("~0", "~"), {}) if isinstance(target, dict) else {}
            if is_array(target, depth + 1):
                return True
        return any(is_array(item, depth + 1) for key in ("anyOf", "oneOf") for item in schema.get(key, []))

    return is_array(root)


def generate_and_encode_tts(text, voice_path, embedding_path=None):
    """TTSを生成してBase64文字列で返す。
    embedding_path が指定されAPIが対応している場合はSpeaker Embeddingを使用する。
    tts_client はスレッドセーフでないため tts_generate_lock で排他制御する。"""
    tts = get_tts_client()
    if tts is None:
        return None

    with tts_generate_lock:
        try:
            settings = load_settings()
            prefer_embedding = settings.get("prefer_speaker_embedding", True)
            # Embeddingを使うか判定: API対応 かつ Embeddingあり かつ (優先設定ON または 音声なし)
            use_embedding = (
                tts_supports_embedding
                and embedding_path is not None
                and (prefer_embedding or voice_path is None)
            )

            print(f"[TTS] generate start: use_embedding={use_embedding} voice_path={voice_path!r} text={text[:20]!r}")
            ver_key = tts_api_version if tts_api_version in DEFAULT_TTS_CONFIG else "v2"
            send_keys = {"v2": _TTS_V2_SEND_KEYS, "v3": _TTS_V3_SEND_KEYS, "v4": _TTS_V4_SEND_KEYS}[ver_key]
            ver_cfg   = TTS_CONFIG.get(ver_key, {})
            filtered_config = {k: v for k, v in ver_cfg.items() if k in send_keys}
            checkpoint = ver_cfg.get(f"checkpoint_{ver_key}", "")
            if ver_key != "v4" or checkpoint:
                filtered_config["checkpoint"] = checkpoint
            if ver_key == "v4":
                # Textbox steps accept blank for model-specific defaults, including MF.
                steps = filtered_config.get("num_steps", "")
                filtered_config["num_steps"] = "" if steps is None else str(steps)
                endpoint = tts._info.get("named_endpoints", {}).get("/_run_generation", {})
                supported = {p.get("parameter_name") for p in endpoint.get("parameters", [])}
                filtered_config = {k: v for k, v in filtered_config.items() if k in supported}

            uploaded_audio = None
            uploaded_embedding = None

            if use_embedding:
                if not os.path.exists(embedding_path):
                    print(f"[TTS] WARNING: embedding_path が見つかりません: {embedding_path}")
                    use_embedding = False
                else:
                    uploaded_embedding = _handle_file_with_retry(embedding_path, "embedding")
                    if uploaded_embedding is None:
                        print(f"[TTS] WARNING: Embedding読み込み失敗。音声ファイルにフォールバック: {embedding_path}")
                        use_embedding = False

            if not use_embedding and voice_path:
                if not os.path.exists(voice_path):
                    print(f"[TTS] WARNING: voice_path が見つかりません: {voice_path}")
                else:
                    uploaded_audio = _handle_file_with_retry(voice_path, "audio")
                    if uploaded_audio is None:
                        print(f"[TTS] WARNING: リファレンス音声の読み込みに失敗。音声なしで生成します: {voice_path}")

            if _tts_reference_is_multiple(tts):
                uploaded_audio = [uploaded_audio] if uploaded_audio is not None else []
            predict_kwargs = {**filtered_config, "text": text, "api_name": "/_run_generation",
                              "uploaded_audio": uploaded_audio}
            if tts_supports_embedding:
                predict_kwargs["uploaded_speaker_embedding"] = uploaded_embedding
            result = tts.predict(**predict_kwargs)

            res_data = result[0]
            gen_path = res_data["value"] if isinstance(res_data, dict) and "value" in res_data else res_data

            if gen_path and os.path.exists(gen_path):
                with open(gen_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                try:
                    os.unlink(gen_path)
                except OSError:
                    pass
                return encoded
        except Exception as e:
            print(f"TTS Error: {e}")
    return None


def enqueue_tts(sentence: str, voice_path: str, req_id: str, embedding_path=None):
    """キューに積む前に queued カウンターをインクリメントする（必ずペアで使う）"""
    with finished_audios_lock:
        if req_id in finished_audios:
            finished_audios[req_id]["queued"] += 1
    generate_queue.put((sentence, voice_path, embedding_path, req_id))


EMOJI_PAT = re.compile(
    r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FFFF\U00002300-\U000023FF]+'
)

def clean_for_tts(text: str) -> str:
    """括弧内を除去するのみ。絵文字は元の位置のまま残す。
    irodori-TTSは文末の絵文字に反応するため、位置を変えずそのまま渡す。"""
    s = re.sub(r'[\(（].*?[\)）]', '', text)
    return s.strip()


def stream_llm_with_tts(messages: list, tts_enabled: bool, memorize_user_text: str | None = None):
    req_id = f"req_{int(time.time() * 1000)}_{threading.get_ident() % 10000}"

    with finished_audios_lock:
        finished_audios[req_id] = {"audios": [], "queued": 0, "completed": 0}

    settings = load_settings()
    tts_batch_size: int = settings.get("tts_batch_size", 0)  # 0=すべて
    no_voice_mode: str  = settings.get("no_voice_mode", "off")

    has_embedding = state.selected.get("has_embedding", False)
    # TTS使用可否: 音声/Embeddingありキャラ、またはデフォルト音声モードで音声なしキャラ
    tts_available = (
        state.selected.get("has_voice")
        or (has_embedding and tts_supports_embedding)
        or no_voice_mode == "default_voice"
    )
    # デフォルト音声モード時は voice_path=None でTTSを呼ぶ
    tts_voice_path = state.selected.get("voice_path") if state.selected.get("has_voice") else None
    tts_embedding_path = state.selected.get("embedding_path") if has_embedding else None

    full_answer = ""
    llm_buffer = ""
    # 句読点・感嘆符の後で分割。直後の絵文字は前トークンに含めるため後続絵文字をバッファへ残す
    split_pat = re.compile(r'(?<=[。！？\?!])(?!\s*' + EMOJI_PAT.pattern + r')')

    # バッチ用: 確定した文のリスト
    pending_sentences: list[str] = []
    enqueued_count: int = 0  # バッチ制御カウンタ

    def flush_audios():
        with finished_audios_lock:
            audios = finished_audios[req_id]["audios"]
            while audios:
                yield f"data: {json.dumps({'audio': audios.pop(0)})}\n\n"

    def maybe_enqueue_batch():
        """pending_sentences を tts_batch_size 件ずつ結合してキューに積む。
        tts_batch_size=0 の場合はストリーム中はエンキューせず最後にまとめて送る。"""
        nonlocal pending_sentences, enqueued_count
        if tts_batch_size == 0:
            return  # 0=すべてまとめて → ストリーム中は積まない
        batch = tts_batch_size
        while len(pending_sentences) >= batch:
            chunk_sentences = pending_sentences[:batch]
            pending_sentences = pending_sentences[batch:]
            combined = "".join(chunk_sentences)
            s = clean_for_tts(combined)
            if len(s) > 1:
                enqueue_tts(s, tts_voice_path, req_id, tts_embedding_path)
                enqueued_count += 1

    try:
        model_name = LLM_MODEL or "local-model"
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_answer += delta
            llm_buffer += delta

            yield f"data: {json.dumps({'token': delta})}\n\n"

            if tts_enabled and tts_available:
                parts = split_pat.split(llm_buffer)
                if len(parts) > 1:
                    # 絵文字が続く場合は次のパーツの先頭絵文字を前のパーツに結合
                    merged: list[str] = []
                    for i, part in enumerate(parts[:-1]):
                        # スペースを挟んだ絵文字も前の文に結合する
                        suffix_match = re.match(r'\s*' + EMOJI_PAT.pattern, parts[i + 1]) if i + 1 < len(parts) else None
                        if suffix_match and suffix_match.group().strip():
                            part = part + suffix_match.group()
                            # 次パーツから絵文字prefix を削除
                            parts[i + 1] = parts[i + 1][suffix_match.end():]
                        merged.append(part)
                    for sentence in merged:
                        if sentence.strip():
                            pending_sentences.append(sentence)
                    llm_buffer = parts[-1]
                    maybe_enqueue_batch()

            # LLM生成中に出来上がった音声はSSEでそのまま流す
            yield from flush_audios()

        # LLM終了後の残りバッファをキューへ
        if tts_enabled and tts_available:
            if llm_buffer.strip():
                pending_sentences.append(llm_buffer)
            if pending_sentences:
                if tts_batch_size == 0:
                    # 0=すべてまとめて → 全文を1リクエストで送る
                    combined = "".join(pending_sentences)
                    pending_sentences = []
                    s = clean_for_tts(combined)
                    if len(s) > 1:
                        enqueue_tts(s, tts_voice_path, req_id, tts_embedding_path)
                else:
                    # 残りをバッチサイズで処理（端数は1つにまとめる）
                    while len(pending_sentences) >= tts_batch_size:
                        chunk = pending_sentences[:tts_batch_size]
                        pending_sentences = pending_sentences[tts_batch_size:]
                        combined = "".join(chunk)
                        s = clean_for_tts(combined)
                        if len(s) > 1:
                            enqueue_tts(s, tts_voice_path, req_id, tts_embedding_path)
                    # 端数
                    if pending_sentences:
                        combined = "".join(pending_sentences)
                        pending_sentences = []
                        s = clean_for_tts(combined)
                        if len(s) > 1:
                            enqueue_tts(s, tts_voice_path, req_id, tts_embedding_path)

        state.memory.append({"role": "assistant", "content": full_answer})
        if memorize_user_text is not None:
            _record_memory_turn(memorize_user_text, full_answer, settings)

        # LLM完了を即通知。残音声があれば req_id をフロントに渡してポーリングさせる
        with finished_audios_lock:
            q = finished_audios[req_id]["queued"]
        has_pending = tts_enabled and tts_available and q > 0
        yield f"data: {json.dumps({'done': True, 'req_id': req_id if has_pending else None})}\n\n"

        # req_id を返した場合はポーリング側が解放するので、ここでは pop しない
        if not has_pending:
            with finished_audios_lock:
                finished_audios.pop(req_id, None)

    except Exception as e:
        import openai as _openai
        err_str = str(e)
        # エラーの種類に応じてフレンドリーメッセージを選択
        if isinstance(e, _openai.APIConnectionError):
            friendly = "LLMサーバが動いてないみたい"
        elif isinstance(e, _openai.APIStatusError):
            body = err_str.lower()
            if "model" in body or e.status_code in (503, 404):
                friendly = "モデルがロードされていないみたい"
            else:
                friendly = "何かエラーが起こったみたい"
        else:
            friendly = "何かエラーが起こったみたい"
        # フレンドリーメッセージをアシスタントの返答として送出
        yield f"data: {json.dumps({'token': friendly})}\n\n"
        # TTS が有効ならフレンドリーメッセージも発声させる
        if tts_enabled and tts_available:
            enqueue_tts(friendly, tts_voice_path, req_id, tts_embedding_path)
            yield f"data: {json.dumps({'done': True, 'req_id': req_id, 'error_detail': err_str})}\n\n"
            # finished_audios の解放はポーリング側（audio_poll）に任せる
        else:
            yield f"data: {json.dumps({'done': True, 'req_id': None, 'error_detail': err_str})}\n\n"
            with finished_audios_lock:
                finished_audios.pop(req_id, None)


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")

    token = _reserve_normal_generation(request)
    user_text = req.message if req.message.strip() else " "
    state.memory.append({"role": "user", "content": user_text})

    try:
        settings = load_settings()
        epoch, revision = _conversation_snapshot()
        stream = stream_llm_with_tts(
            _build_llm_messages(settings), req.tts_enabled,
            memorize_user_text=user_text if req.memorize else None,
        )
        return StreamingResponse(
            _tracked_normal_stream(stream, token),
            media_type="text/event-stream",
            headers={"X-Conversation-Epoch": epoch, "X-Normal-Revision": str(revision)},
        )
    except Exception:
        _release_normal_generation(token)
        raise


@app.get("/initial_greeting")
def initial_greeting(request: Request, tts_enabled: bool = True):
    if not state.selected:
        raise HTTPException(400, "Character not selected")

    token = _reserve_normal_generation(request)
    try:
        settings = load_settings()
        state.memory.append({"role": "user", "content": _greeting_instruction(settings)})
        epoch, revision = _conversation_snapshot()
        return StreamingResponse(
            _tracked_normal_stream(stream_llm_with_tts(_build_llm_messages(settings), tts_enabled), token),
            media_type="text/event-stream",
            headers={"X-Conversation-Epoch": epoch, "X-Normal-Revision": str(revision)},
        )
    except Exception:
        _release_normal_generation(token)
        raise


@app.get("/audio_poll/{req_id}", name="audio_poll_get")
def audio_poll(req_id: str):
    """
    バックグラウンドTTS生成の進捗をポーリングするエンドポイント。
    - audios: 今この瞬間に出来上がっている音声(base64)リスト
    - finished: queued==completed で全音声生成完了
    """
    with finished_audios_lock:
        entry = finished_audios.get(req_id)
        if entry is None:
            return JSONResponse({"audios": [], "finished": True})

        audios = entry["audios"][:]
        entry["audios"].clear()
        q = entry["queued"]
        c = entry["completed"]
        finished = (q > 0 and q == c)

    if finished:
        # 全完了したらメモリ解放
        with finished_audios_lock:
            finished_audios.pop(req_id, None)
        print(f"[TTS] All {q} audio(s) polled and released for {req_id}")

    return JSONResponse({"audios": audios, "finished": finished})


@app.on_event("startup")
def on_startup():
    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=_game_lease_reaper, daemon=True).start()


# --- 自動応答: ユーザー役返答生成 ---
_DEFAULT_USER_REPLY_PROMPT = (
    "あなたはユーザー役のライターです。"
    "与えられた「キャラクターの最新発言」に対して、"
    "ユーザーが自然に返しそうな一言を1〜2文で作ってください。\n"
    "発言内容・質問・話題に具体的に反応すること。"
    "キャラの口調は真似しない。話し言葉で書く。"
    "返答テキストのみ出力し、前置きや説明は一切不要。"
)

@app.post("/user_reply_gen")
def user_reply_gen(request: Request):
    """自動応答用: ユーザー役の返答テキストを生成して返す"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")

    token = _reserve_normal_generation(request, bump_revision=False)
    try:
        return _user_reply_gen_reserved()
    finally:
        _release_normal_generation(token)


def _user_reply_gen_reserved():
    settings = load_settings()
    custom_prompt = (settings.get("auto_reply_user_prompt") or "").strip()
    user_prompt = (
        f"{_DEFAULT_USER_REPLY_PROMPT}\n\n{custom_prompt}" if custom_prompt
        else _DEFAULT_USER_REPLY_PROMPT
    )

    # キャラクターの最新発言を取得
    last_char_msg = ""
    for m in reversed(list(state.memory)):
        if m.get("role") == "assistant":
            last_char_msg = m.get("content", "").strip()
            break

    if not last_char_msg:
        raise HTTPException(400, "キャラクターの発言履歴がありません")

    # ユーザー役プロンプトのみを system に設定する。
    # キャラのシステムプロンプトを含めると LLM がキャラ役を優先して
    # オウム返しになるため、意図的に除外している。
    # last_char_msg の文脈だけで十分なユーザー返答が生成できる。
    messages = [
        {"role": "system", "content": user_prompt},
        {"role": "user",   "content": f"キャラクターの最新発言:\n{last_char_msg}\n---\nユーザーの返答:"},
    ]

    try:
        model_name = LLM_MODEL or "local-model"
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(500, "自動応答の生成に失敗しました") from e


# --- 直接TTS (LLMを通さずTTSだけ実行) ---
_DEFAULT_SUMMARY_PROMPT = (
    "以下のWebページ本文を日本語で読み上げ用に要約してください。\n"
    "聴いて内容が理解できるよう、重要な情報（主題・主要な事実・結論）を漏らさず、"
    "自然な話し言葉の文章でまとめてください。\n"
    "箇条書きは使わず、段落を意識した読みやすい文体にしてください。\n"
    "分量の目安: 400〜800文字程度（元の内容が短い場合はそれに応じて短くして構いません）。"
)

class ReadUrlRequest(BaseModel):
    url:            str
    summarize:      bool = False
    summary_prompt: str  = ""   # 空=デフォルト使用


@app.post("/read_url")
def read_url(req: ReadUrlRequest):
    """URLの本文を抽出し、オプションでLLM要約して返す"""
    import trafilatura, json as _json, requests as _req
    from charset_normalizer import from_bytes as _from_bytes

    # フェッチ（charset-normalizer でエンコーディングを精密検出）
    try:
        r = _req.get(
            req.url, timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BridgeTTS/1.0)"},
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"URLの取得に失敗しました: {e}")

    import re as _re

    def _detect_encoding(resp) -> str:
        """HTTP ヘッダー → meta charset → charset-normalizer の順で検出"""
        # 1. HTTP Content-Type ヘッダーの charset
        ct = resp.headers.get("content-type", "")
        m = _re.search(r"charset=([^\s;]+)", ct, _re.IGNORECASE)
        enc = m.group(1).strip() if m else None

        # 2. HTML 先頭 2KB の meta charset
        if not enc:
            head = resp.content[:2048]
            m = _re.search(rb'charset=["\']?([^"\';\s>]+)', head, _re.IGNORECASE)
            enc = m.group(1).decode("ascii", errors="ignore").strip() if m else None

        if enc:
            # Shift-JIS 系のバリエーションを cp932 に統一
            enc_lower = enc.lower().replace("-", "").replace("_", "")
            if enc_lower in ("shiftjis", "sjis", "xsjis", "windows31j",
                             "cp932", "mskanji", "csshiftjis"):
                return "cp932"
            return enc

        # 3. charset-normalizer による統計的推定
        best = _from_bytes(resp.content).best()
        return best.first().encoding if best else "utf-8"

    enc = _detect_encoding(r)
    try:
        html_str = r.content.decode(enc, errors="replace")
    except (LookupError, Exception):
        html_str = r.content.decode("utf-8", errors="replace")

    # 本文抽出（メタデータ付き JSON 形式）
    meta_str = trafilatura.extract(
        html_str,
        output_format="json",
        include_links=False,
        include_tables=False,
        include_images=False,
    )
    if meta_str:
        meta_obj = _json.loads(meta_str)
        raw_text = meta_obj.get("text") or ""
        title    = meta_obj.get("title") or ""
    else:
        raw_text = ""
        title    = ""

    if not raw_text.strip():
        raise HTTPException(400, "本文の抽出に失敗しました（ペイウォール・JS必須のページは非対応）")

    text = raw_text.strip()

    # LLM要約（オプション）
    if req.summarize:
        excerpt = text[:6000]  # トークン上限対策
        prompt_text  = req.summary_prompt.strip() if req.summary_prompt.strip() else _DEFAULT_SUMMARY_PROMPT
        summary_user = f"{prompt_text}\n\n【本文】\n{excerpt}"
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL or "local-model",
                messages=[{"role": "user", "content": summary_user}],
                stream=False,
            )
            text = (resp.choices[0].message.content or "").strip() or text
        except Exception:
            pass  # 要約失敗時は原文をそのまま使用

    return {"title": title, "text": text, "summarized": req.summarize}


_DEFAULT_TEXT_SUMMARY_PROMPT = (
    "以下のテキストを日本語で読み上げ用に要約してください。\n"
    "聴いて内容が理解できるよう、重要な情報（主題・主要な事実・結論）を漏らさず、"
    "自然な話し言葉の文章でまとめてください。\n"
    "箇条書きは使わず、段落を意識した読みやすい文体にしてください。\n"
    "分量の目安: 400〜800文字程度（元の内容が短い場合はそれに応じて短くして構いません）。"
)

class SummarizeTextRequest(BaseModel):
    text:           str
    summary_prompt: str = ""

@app.post("/summarize_text")
def summarize_text_endpoint(req: SummarizeTextRequest):
    """プレーンテキストをLLMで要約して返す"""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "テキストが空です")

    excerpt = text[:6000]
    prompt_text = req.summary_prompt.strip() if req.summary_prompt.strip() else _DEFAULT_TEXT_SUMMARY_PROMPT
    summary_user = f"{prompt_text}\n\n【本文】\n{excerpt}"
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL or "local-model",
            messages=[{"role": "user", "content": summary_user}],
            stream=False,
        )
        result = (resp.choices[0].message.content or "").strip()
        if not result:
            raise HTTPException(500, "要約の生成に失敗しました")
        return {"text": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"要約エラー: {e}")


class DirectTTSRequest(BaseModel):
    text: str

@app.post("/tts_direct")
def tts_direct(req: DirectTTSRequest, request: Request):
    token = _reserve_normal_generation(request, bump_revision=False)
    try:
        return _tts_direct_reserved(req)
    finally:
        _release_normal_generation(token)


def _tts_direct_reserved(req: DirectTTSRequest):
    """入力テキストをそのままTTSに渡して音声を返す"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")
    voice_path = state.selected.get("voice_path")
    embedding_path = state.selected.get("embedding_path")
    settings = load_settings()
    no_voice_mode = settings.get("no_voice_mode", "off")
    has_tts_source = (
        voice_path
        or (embedding_path and tts_supports_embedding)
        or no_voice_mode == "default_voice"
    )
    if not has_tts_source:
        raise HTTPException(400, "音声ファイルが設定されていません")

    text = clean_for_tts(req.text.strip())
    if not text:
        raise HTTPException(400, "テキストが空です")

    audio_b64 = generate_and_encode_tts(text, voice_path, embedding_path)
    if audio_b64 is None:
        raise HTTPException(500, "TTS生成に失敗しました")

    return JSONResponse({"audio": audio_b64})


@app.delete("/audio_poll/{req_id}")
def audio_poll_cancel(req_id: str):
    with finished_audios_lock:
        finished_audios.pop(req_id, None)
    return JSONResponse({"status": "cancelled"})

@app.get("/settings")
def get_settings():
    """設定ファイルを読み込んで返す"""
    return JSONResponse(load_settings())


@app.post("/settings")
def post_settings(data: dict = Body(...)):
    """設定を保存する（ロック付き read-modify-write でレースコンディションを防ぐ）"""
    previous_auto_save = True
    updated_auto_save = True
    with _settings_lock:
        current = load_settings()
        previous_auto_save = current.get("log_auto_save", True) is not False
        current.update(data)
        updated_auto_save = current.get("log_auto_save", True) is not False
        save_settings(current)
    if not previous_auto_save and updated_auto_save:
        _game_engine.save_current_log()
    return JSONResponse({"status": "ok"})


# ══════════════════════════════════════════════════════
#  ゲームモード API — 通常チャット・通常記憶から独立
# ══════════════════════════════════════════════════════
class _StrictGameModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GameStartRequest(_StrictGameModel):
    page_id: str
    conversation_epoch: str
    normal_revision: int
    request_id: str


class GameInput(_StrictGameModel):
    kind: str
    interaction_id: str
    text: str | None = None
    choice_id: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self):
        if len(self.interaction_id) > 128:
            raise ValueError("interaction_id is too long")
        if self.kind == "text":
            if self.text is None or self.choice_id is not None or len(self.text) > 2000:
                raise ValueError("invalid text input")
        elif self.kind == "choice":
            if not self.choice_id or self.text is not None or len(self.choice_id) > 128:
                raise ValueError("invalid choice input")
        else:
            raise ValueError("invalid input kind")
        return self


class GameTurnRequest(_StrictGameModel):
    page_id: str
    conversation_epoch: str
    revision: int
    request_id: str
    input: GameInput


class GameSuspendRequest(_StrictGameModel):
    page_id: str
    revision: int
    request_id: str


class GameResumeRequest(_StrictGameModel):
    page_id: str
    conversation_epoch: str
    revision: int
    request_id: str


class GameCloseRequest(_StrictGameModel):
    page_id: str
    request_id: str
    reason: str = "user"


class GameClosePageRequest(_StrictGameModel):
    page_id: str
    request_id: str


class GameLeaseRequest(_StrictGameModel):
    page_id: str


class GameAudioRequest(_StrictGameModel):
    page_id: str
    event_id: str
    revision: int
    request_id: str
    retry: bool = False


class GameLogExportRequest(_StrictGameModel):
    page_id: str


class ConversationViewResetRequest(_StrictGameModel):
    page_id: str
    conversation_epoch: str
    normal_revision: int
    request_id: str


def _capture_game_seed() -> dict:
    if not state.selected or not state.memory:
        raise HTTPException(400, detail={"code": "CHARACTER_NOT_SELECTED"})
    settings = load_settings()
    # This renders pre-game long-term memory only when memory_enabled is on.
    system_messages = _build_llm_messages(settings)
    system_prompt = system_messages[0]["content"] if system_messages else state.memory[0]["content"]
    greeting = _greeting_instruction(settings)
    history = [
        {"role": item.get("role"), "content": item.get("content", "")}
        for item in state.memory[1:]
        if item.get("role") in {"user", "assistant"}
        and item.get("content")
        and not (item.get("role") == "user" and item.get("content") == greeting)
    ]
    history = history[-24:]
    while len(json.dumps(history, ensure_ascii=False).encode("utf-8")) > 24_000 and len(history) > 2:
        del history[:2]
    long_memory = load_memory(state.selected["name"]) if settings.get("memory_enabled", True) else None
    selected = copy.deepcopy(state.selected)
    has_embedding = bool(selected.get("has_embedding") and tts_supports_embedding)
    has_tts = bool(selected.get("has_voice") or has_embedding or settings.get("no_voice_mode") == "default_voice")
    return {
        "character_id": os.path.abspath(selected.get("prompt", selected.get("name", ""))),
        "character_name": selected.get("name", ""),
        "system_prompt": system_prompt,
        "normal_history": history,
        "long_term_memory": long_memory,
        "memory_enabled": bool(settings.get("memory_enabled", True)),
        "started_context": {"time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "voice": {
            "voice_path": selected.get("voice_path") if selected.get("has_voice") else None,
            "embedding_path": selected.get("embedding_path") if has_embedding else None,
            "tts_available": has_tts,
        },
    }


def _game_lease_reaper():
    while True:
        time.sleep(60)
        _game_engine.cleanup_expired()


@app.get("/conversation/status")
def conversation_status(request: Request):
    page_id = request.headers.get("X-Bridge-Page-Id")
    activity = _game_engine.activity()
    with _conversation_gate:
        return {
            "conversation_epoch": _conversation_epoch,
            "normal_revision": _normal_revision,
            "busy": bool(_normal_generations) or activity in {"starting", "resolving"},
            "game_active": activity in {"starting", "active", "resolving"},
            "game_suspended": activity == "suspended" and bool(_game_engine.current(page_id or "")),
        }


@app.post("/conversation/view-reset")
def conversation_view_reset(req: ConversationViewResetRequest):
    now = time.time()
    for key, (_, _, expiry) in list(_view_reset_records.items()):
        if expiry <= now:
            _view_reset_records.pop(key, None)
    record_key = (req.page_id, req.request_id)
    payload_hash = json.dumps(req.model_dump(), sort_keys=True, ensure_ascii=False)
    with _conversation_gate:
        prior = _view_reset_records.get(record_key)
        if prior:
            if prior[0] != payload_hash:
                raise HTTPException(409, detail={"code": "REQUEST_ID_REUSED"})
            return prior[1]
        if req.conversation_epoch != _conversation_epoch or req.normal_revision != _normal_revision:
            raise HTTPException(409, detail={"code": "STALE_CONVERSATION"})
        result = _conversation_transition(page_id=req.page_id, epoch=req.conversation_epoch, view_reset=True)
        _view_reset_records[record_key] = (payload_hash, result, now + 300)
        return result


def _game_stream_response(operation, phase: str):
    """Stream public previews; generation survives disconnect for idempotent recovery."""
    events = queue.Queue(maxsize=128)
    disconnected = threading.Event()

    def emit(event):
        if disconnected.is_set():
            return
        # Previews are full text snapshots, so a slow reader can skip old snapshots.
        if events.full():
            try:
                events.get_nowait()
            except queue.Empty:
                pass
        events.put_nowait(event)

    def run():
        try:
            emit({"type": "phase", "phase": phase})
            result = operation(emit)
            emit({"type": "result", "result": result})
        except GameError as error:
            emit({"type": "error", "code": error.code, "status": error.status_code})
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, dict) else {}
            emit({"type": "error", "code": detail.get("code", "INVALID_REQUEST"), "status": error.status_code})
        except Exception as error:
            print(f"[Game] Streaming request failed: {type(error).__name__}")
            emit({"type": "error", "code": "INTERNAL_ERROR", "status": 500})

    def body():
        threading.Thread(target=run, daemon=True).start()
        try:
            while True:
                try:
                    event = events.get(timeout=10)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                if event["type"] in {"result", "error"}:
                    return
        finally:
            disconnected.set()

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/game/start")
def game_start(req: GameStartRequest, stream: bool = False):
    def operation(emit):
        return _game_engine.start(
            page_id=req.page_id, conversation_epoch=req.conversation_epoch,
            normal_revision=req.normal_revision, request_id=req.request_id,
            seed=_capture_game_seed, on_event=emit,
        )
    return _game_stream_response(operation, "opening") if stream else operation(None)


@app.post("/game/{game_id}/turn")
def game_turn(game_id: str, req: GameTurnRequest, stream: bool = False):
    def operation(emit):
        return _game_engine.submit(
            game_id, page_id=req.page_id, conversation_epoch=req.conversation_epoch,
            revision=req.revision, request_id=req.request_id,
            request_input=req.input.model_dump(exclude_none=True), on_event=emit,
        )
    return _game_stream_response(operation, "resolve") if stream else operation(None)


@app.get("/game/{game_id}")
def game_view(game_id: str, page_id: str):
    return _game_engine.view(game_id, page_id)


@app.get("/game/{game_id}/requests/{request_id}")
def game_request_result(game_id: str, request_id: str, page_id: str):
    return _game_engine.request_result(game_id, page_id, request_id)


@app.post("/game/{game_id}/suspend")
def game_suspend(game_id: str, req: GameSuspendRequest):
    return _game_engine.suspend(game_id, page_id=req.page_id, revision=req.revision, request_id=req.request_id)


@app.post("/game/{game_id}/resume")
def game_resume(game_id: str, req: GameResumeRequest):
    return _game_engine.resume(
        game_id, page_id=req.page_id, conversation_epoch=req.conversation_epoch,
        revision=req.revision, request_id=req.request_id,
    )


@app.post("/game/{game_id}/close", status_code=204)
def game_close(game_id: str, req: GameCloseRequest):
    _game_engine.close(game_id, page_id=req.page_id, request_id=req.request_id)
    return Response(status_code=204)


@app.post("/game/close-page", status_code=204)
def game_close_page(req: GameClosePageRequest):
    _game_engine.close_page(req.page_id)
    return Response(status_code=204)


@app.post("/game/{game_id}/lease", status_code=204)
def game_lease(game_id: str, req: GameLeaseRequest):
    _game_engine.lease(game_id, req.page_id)
    return Response(status_code=204)


@app.post("/game/{game_id}/audio")
def game_audio(game_id: str, req: GameAudioRequest):
    if not load_settings().get("tts_enabled", True):
        raise HTTPException(400, detail={"code": "TTS_DISABLED"})
    record = _game_engine.audio_payload(
        game_id, req.page_id, req.event_id, req.revision, req.request_id, retry=req.retry,
    )
    req_id = record["audio_req_id"]
    if not record.get("already_started"):
        with finished_audios_lock:
            finished_audios.setdefault(req_id, {"audios": [], "queued": 0, "completed": 0})
        try:
            enqueue_tts(record["text"], record["voice_path"], req_id, record["embedding_path"])
        except Exception as exc:
            with finished_audios_lock:
                finished_audios.pop(req_id, None)
            raise HTTPException(502, detail={"code": "TTS_FAILED"}) from exc
    return {"req_id": req_id}


@app.post("/game/{game_id}/log/export")
def game_log_export(game_id: str, req: GameLogExportRequest):
    content, filename = _game_engine.export_log(game_id, req.page_id)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- プロンプトファイル一覧 ---
@app.get("/prompts")
def list_prompts():
    """prompts/ フォルダ内の .txt ファイル一覧を返す。"""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    files = sorted([
        f for f in os.listdir(PROMPTS_DIR)
        if f.endswith(".txt") and os.path.isfile(os.path.join(PROMPTS_DIR, f))
    ])
    return JSONResponse(files)


@app.get("/prompts/{filename}")
def get_prompt(filename: str):
    """プロンプトファイルの内容を返す。"""
    # パストラバーサル防止
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    fpath = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")
    with open(fpath, "r", encoding="utf-8") as f:
        content_text = f.read()
    return JSONResponse({"filename": filename, "content": content_text})


# ══════════════════════════════════════════════════════
#  TTS設定 API
# ══════════════════════════════════════════════════════

@app.get("/tts_config")
def get_tts_config_api(version: str = ""):
    """TTS詳細設定を返す。version="v2"|"v3"|"v4" で指定（省略時は全体）"""
    cfg = load_tts_config()
    if version in DEFAULT_TTS_CONFIG:
        return JSONResponse(cfg.get(version, {}))
    return JSONResponse(cfg, headers={"X-TTS-API-Version": tts_api_version if tts_client is not None else ""})


def _validate_tts_config_patch(version: str, patch: dict) -> dict:
    if not isinstance(patch, dict):
        raise HTTPException(422, "TTS settings must be an object")
    patch = dict(patch)
    checkpoint_key = f"checkpoint_{version}"
    if checkpoint_key in patch and not isinstance(patch[checkpoint_key], str):
        raise HTTPException(422, "Checkpoint must be a string")
    if "num_steps" in patch:
        steps = patch["num_steps"]
        if steps == "" and version == "v4":
            pass
        elif isinstance(steps, bool) or not re.fullmatch(r"[1-9]\d*", str(steps)):
            raise HTTPException(422, "Steps must be a positive integer (or blank for v4)")
        else:
            patch["num_steps"] = int(steps)
    if "t_schedule_mode" in patch and patch["t_schedule_mode"] not in ("linear", "sway"):
        raise HTTPException(422, "Unknown sampling schedule")
    if "sway_coeff" in patch:
        coefficient = patch["sway_coeff"]
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or not -1 <= coefficient <= 1.5:
            raise HTTPException(422, "Sway coefficient must be between -1 and 1.5")
    if "model_presets" in patch:
        presets = patch["model_presets"]
        if not isinstance(presets, list):
            raise HTTPException(422, "Model presets must be a list")
        cleaned = []
        names, checkpoints = set(), set()
        for entry in presets:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not isinstance(entry.get("checkpoint"), str):
                raise HTTPException(422, "Model presets require a name and checkpoint")
            name, checkpoint = entry["name"].strip(), entry["checkpoint"].strip()
            if not 1 <= len(name) <= 80 or not 1 <= len(checkpoint) <= 2048 or name in names or checkpoint in checkpoints:
                raise HTTPException(422, "Model presets must have unique, nonempty names and checkpoints")
            names.add(name)
            checkpoints.add(checkpoint)
            cleaned.append({"name": name, "checkpoint": checkpoint})
        patch["model_presets"] = cleaned
    return patch

@app.post("/tts_config")
def post_tts_config_api(data: dict = Body(...)):
    """TTS詳細設定を保存する。
    {"version": "v2", ...params} または {"v2": {...}, "v3": {...}} 形式を受け付ける"""
    global TTS_CONFIG
    current = load_tts_config()

    if "version" in data:
        # バージョン指定形式: {"version": "v2", "num_steps": 10, ...}
        ver = data["version"]
        if ver not in DEFAULT_TTS_CONFIG:
            raise HTTPException(422, "Unknown TTS API profile")
        current[ver].update(_validate_tts_config_patch(ver, {k: v for k, v in data.items() if k != "version"}))
    elif any(version in data for version in DEFAULT_TTS_CONFIG):
        # 全体形式: {"v2": {...}, "v3": {...}}
        for ver in DEFAULT_TTS_CONFIG:
            if ver in data:
                current[ver].update(_validate_tts_config_patch(ver, data[ver]))
    else:
        # バージョン指定なし: 現在の tts_api_version に適用
        ver = tts_api_version if tts_api_version in DEFAULT_TTS_CONFIG else "v2"
        current[ver].update(_validate_tts_config_patch(ver, data))

    save_tts_config(current)
    TTS_CONFIG = current
    return JSONResponse({"status": "ok"})


# ══════════════════════════════════════════════════════
#  吹き出し再生成 API (文章分割 + バッチTTS)
# ══════════════════════════════════════════════════════

class TtsRegenRequest(BaseModel):
    text: str

@app.post("/tts_regen")
def tts_regen(req: TtsRegenRequest):
    """テキストをsettingsの文章数に従って分割・バッチキューに積み req_id を返す。
    フロントエンドは /audio_poll/{req_id} でポーリングして音声を受け取る。"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")
    voice_path = state.selected.get("voice_path")
    embedding_path = state.selected.get("embedding_path")
    settings = load_settings()
    no_voice_mode = settings.get("no_voice_mode", "off")
    has_tts_source = (
        voice_path
        or (embedding_path and tts_supports_embedding)
        or no_voice_mode == "default_voice"
    )
    if not has_tts_source:
        raise HTTPException(400, "音声ファイルが設定されていません")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "テキストが空です")

    settings = load_settings()
    tts_batch_size: int = settings.get("tts_batch_size", 0)

    req_id = f"regen_{int(time.time() * 1000)}"
    with finished_audios_lock:
        finished_audios[req_id] = {"audios": [], "queued": 0, "completed": 0}

    # ── 文章分割 (stream_llm_with_tts と同じロジック) ──
    split_pat = re.compile(r'(?<=[。！？\?!])(?!\s*' + EMOJI_PAT.pattern + r')')
    parts = split_pat.split(text)
    sentences: list[str] = []
    if len(parts) > 1:
        merged: list[str] = []
        for i, part in enumerate(parts[:-1]):
            suffix_match = re.match(r'\s*' + EMOJI_PAT.pattern, parts[i + 1]) if i + 1 < len(parts) else None
            if suffix_match and suffix_match.group().strip():
                part = part + suffix_match.group()
                parts[i + 1] = parts[i + 1][suffix_match.end():]
            merged.append(part)
        sentences = [s for s in merged if s.strip()]
        if parts[-1].strip():
            sentences.append(parts[-1])
    else:
        sentences = [text]

    # ── バッチキューに積む ──
    def enqueue_batch(chunk: list[str]):
        combined = "".join(chunk)
        s = clean_for_tts(combined)
        if len(s) > 1:
            enqueue_tts(s, voice_path, req_id, embedding_path)

    if tts_batch_size == 0:
        enqueue_batch(sentences)
    else:
        while len(sentences) >= tts_batch_size:
            enqueue_batch(sentences[:tts_batch_size])
            sentences = sentences[tts_batch_size:]
        if sentences:
            enqueue_batch(sentences)

    with finished_audios_lock:
        queued = finished_audios[req_id]["queued"]

    if queued == 0:
        with finished_audios_lock:
            finished_audios.pop(req_id, None)
        raise HTTPException(500, "TTSキューへの登録に失敗しました")

    return JSONResponse({"req_id": req_id})


# ══════════════════════════════════════════════════════
#  音声エクスポート / ファイルを場所で開く API
# ══════════════════════════════════════════════════════

EXPORTS_DIR = "exports"
LOG_DIR     = "log"
OUTPUT_DIR  = "output"


# ── エクスポート用ヘルパー ────────────────────────────────────────────────

def combine_wavs_bytes(wav_bytes_list: list) -> bytes:
    """複数のWAVバイト列を連結してひとつのWAVバイト列にする"""
    import wave, io
    if not wav_bytes_list:
        return b""
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]
    params = None
    all_frames = []
    for wb in wav_bytes_list:
        with wave.open(io.BytesIO(wb)) as wf:
            if params is None:
                params = wf.getparams()
            all_frames.append(wf.readframes(wf.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        for frames in all_frames:
            wf.writeframes(frames)
    return out.getvalue()


def get_ffmpeg_cmd(settings: dict) -> str | None:
    """設定に基づいて使用するffmpegコマンドを返す。利用不可なら None。"""
    import shutil
    path_enabled = settings.get("ffmpeg_path_enabled", False)
    path_val     = settings.get("ffmpeg_path", "").strip()
    cmd = path_val if (path_enabled and path_val) else "ffmpeg"
    if shutil.which(cmd) or (os.path.isfile(cmd) and os.access(cmd, os.X_OK)):
        return cmd
    return None


def convert_wav_to_mp3(wav_bytes: bytes, ffmpeg_cmd: str) -> bytes:
    """WAVバイト列をMP3バイト列に変換する"""
    result = subprocess.run(
        [ffmpeg_cmd, "-y", "-f", "wav", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-q:a", "2", "-f", "mp3", "pipe:1"],
        input=wav_bytes, capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def make_export_filename(char_name: str, text: str, ext: str) -> str:
    """キャラ名-タイムスタンプ-セリフ冒頭.ext 形式のファイル名を返す"""
    safe_char = re.sub(r'[\\/:*?"<>| ]', '_', char_name)
    head = re.sub(r'[\\/:*?"<>|\s\n\r]', '', text[:15]).strip("_")
    if not head:
        head = "audio"
    timestamp = int(time.time() * 1000)
    return f"{safe_char}-{timestamp}-{head}.{ext}"


def _split_into_batches(text: str, tts_batch_size: int) -> list[str]:
    """tts_batch_size に従ってテキストをバッチリストに分割する（tts_regen と同ロジック）"""
    split_pat = re.compile(r'(?<=[。！？\?!])(?!\s*' + EMOJI_PAT.pattern + r')')
    parts = split_pat.split(text)
    sentences: list[str] = []
    if len(parts) > 1:
        merged: list[str] = []
        for i, part in enumerate(parts[:-1]):
            suffix_match = re.match(r'\s*' + EMOJI_PAT.pattern, parts[i + 1]) if i + 1 < len(parts) else None
            if suffix_match and suffix_match.group().strip():
                part = part + suffix_match.group()
                parts[i + 1] = parts[i + 1][suffix_match.end():]
            merged.append(part)
        sentences = [s for s in merged if s.strip()]
        if parts[-1].strip():
            sentences.append(parts[-1])
    else:
        sentences = [text] if text.strip() else []

    if not sentences:
        return [text] if text.strip() else []

    if tts_batch_size == 0:
        return ["".join(sentences)]

    batches: list[str] = []
    while len(sentences) >= tts_batch_size:
        batches.append("".join(sentences[:tts_batch_size]))
        sentences = sentences[tts_batch_size:]
    if sentences:
        batches.append("".join(sentences))
    return batches


class TtsExportRequest(BaseModel):
    text: str

@app.post("/tts_export")
def tts_export(req: TtsExportRequest):
    """テキストをTTSで生成(バッチ分割→WAV結合→任意でMP3変換)してファイルとして保存。"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")
    voice_path = state.selected.get("voice_path")
    embedding_path = state.selected.get("embedding_path")
    settings = load_settings()
    no_voice_mode = settings.get("no_voice_mode", "off")
    has_tts_source = (
        voice_path
        or (embedding_path and tts_supports_embedding)
        or no_voice_mode == "default_voice"
    )
    if not has_tts_source:
        raise HTTPException(400, "音声ファイルが設定されていません")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "テキストが空です")

    settings      = load_settings()
    tts_batch_size = settings.get("tts_batch_size", 0)

    # バッチ分割して各バッチをTTS生成
    batches = _split_into_batches(text, tts_batch_size)
    wav_bytes_list: list[bytes] = []
    for batch_text in batches:
        s = clean_for_tts(batch_text)
        if len(s) <= 1:
            continue
        audio_b64 = generate_and_encode_tts(s, voice_path, embedding_path)
        if audio_b64 is None:
            raise HTTPException(500, "TTS生成に失敗しました")
        wav_bytes_list.append(base64.b64decode(audio_b64))

    if not wav_bytes_list:
        raise HTTPException(500, "TTS生成に失敗しました")

    # WAV結合
    combined = combine_wavs_bytes(wav_bytes_list)

    # ffmpeg MP3変換（設定が有効かつffmpegが使える場合）
    ext  = "wav"
    mime = "audio/wav"
    ffmpeg_to_mp3 = settings.get("ffmpeg_to_mp3", False)
    ffmpeg_cmd    = get_ffmpeg_cmd(settings)
    if ffmpeg_to_mp3 and ffmpeg_cmd:
        try:
            combined = convert_wav_to_mp3(combined, ffmpeg_cmd)
            ext  = "mp3"
            mime = "audio/mpeg"
        except Exception as e:
            print(f"[ffmpeg] MP3変換失敗、WAVで出力: {e}")

    # 保存
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    char_name = state.selected.get("name", "unknown")
    filename  = make_export_filename(char_name, text, ext)
    filepath  = os.path.join(EXPORTS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(combined)

    audio_b64 = base64.b64encode(combined).decode("utf-8")
    return JSONResponse({
        "audio":    audio_b64,
        "filename": filename,
        "filepath": os.path.abspath(filepath),
        "format":   ext,
    })


@app.get("/check_ffmpeg")
def check_ffmpeg():
    """ffmpegが使用可能か確認する"""
    settings  = load_settings()
    cmd       = get_ffmpeg_cmd(settings)
    return JSONResponse({"available": cmd is not None, "cmd": cmd or ""})


@app.get("/tts_export/{filename}")
def download_tts_export(filename: str):
    """保存済みエクスポートWAVをブラウザにダウンロードさせる"""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    filepath = os.path.join(EXPORTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")
    return FileResponse(filepath, media_type="audio/wav", filename=filename)


class OpenLocationRequest(BaseModel):
    filepath: str

@app.post("/open_file_location")
def open_file_location(req: OpenLocationRequest):
    """サーバーホスト上のOSファイルマネージャーで指定ファイルを選択した状態で開く"""
    filepath = req.filepath
    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")
    try:
        import platform
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(f'explorer /select,"{filepath}"', shell=True)
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", filepath])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(filepath)])
        return JSONResponse({"status": "ok"})
    except Exception as e:
        raise HTTPException(500, f"Failed to open file location: {e}")


# ══════════════════════════════════════════════════════
#  接続設定 API (一時的・再起動で失われる)
# ══════════════════════════════════════════════════════

@app.get("/api_config")
def get_api_config():
    """現在の接続設定を返す（設定ファイルの値 + 環境変数の状態）"""
    settings = load_settings()
    return JSONResponse({
        # LLM設定
        "llm_use_env":     settings.get("llm_use_env", False),
        "llm_preset":      settings.get("llm_preset", "lm-studio"),
        "llm_host":        settings.get("llm_host", "localhost"),
        "llm_port":        settings.get("llm_port", 1234),
        "llm_path":        settings.get("llm_path", "/v1"),
        "llm_api_key":     settings.get("llm_api_key", "lm-studio"),
        "llm_model":       settings.get("llm_model", ""),
        "llm_api_url":     LLM_API_URL,   # 現在実際に使用中のURL
        # TTS設定
        "tts_use_env":     settings.get("tts_use_env", False),
        "tts_host":        settings.get("tts_host", "localhost"),
        "tts_port":        settings.get("tts_port", 7860),
        "tts_api_url":     TTS_API_URL,   # 現在実際に使用中のURL
        # 環境変数の現在値
        "env_llm_api_url": _ENV_LLM_API_URL,
        "env_tts_api_url": _ENV_TTS_API_URL,
        # CHAR_DIR 環境変数の状態（後方互換）
        "char_dir_active": bool(CHAR_DIR_ENV),
        "char_dir_value":  CHAR_DIR_ENV or "",
    })


@app.post("/api_config")
def post_api_config(data: dict = Body(...)):
    """接続設定を変更してファイルに永続化する。"""
    global LLM_API_URL, TTS_API_URL, LLM_API_KEY, LLM_MODEL, client, tts_client

    settings = load_settings()
    changed  = []

    # ── LLM設定 ──
    llm_fields = ["llm_use_env", "llm_preset", "llm_host", "llm_port",
                  "llm_path", "llm_api_key", "llm_model"]
    llm_changed = any(f in data for f in llm_fields)
    if llm_changed:
        for f in llm_fields:
            if f in data:
                settings[f] = data[f]
        new_url = _build_llm_url(settings)
        new_key = (settings.get("llm_api_key") or "lm-studio")
        new_model = (settings.get("llm_model") or "")
        LLM_API_URL = new_url
        LLM_API_KEY = new_key
        LLM_MODEL   = new_model
        client = OpenAI(base_url=LLM_API_URL, api_key=LLM_API_KEY)
        changed.append("llm")
        print(f"[API Config] LLM -> {LLM_API_URL}, key={LLM_API_KEY!r}, model={LLM_MODEL!r}")

    # ── TTS設定 ──
    tts_fields = ["tts_use_env", "tts_host", "tts_port"]
    tts_changed = any(f in data for f in tts_fields)
    if tts_changed:
        for f in tts_fields:
            if f in data:
                settings[f] = data[f]
        new_tts_url = _build_tts_url(settings)
        TTS_API_URL = new_tts_url
        tts_client  = None   # 次回 get_tts_client() 呼び出しで再接続
        changed.append("tts")
        print(f"[API Config] TTS -> {TTS_API_URL}")

    save_settings(settings)
    return JSONResponse({"status": "ok", "changed": changed})


@app.get("/llm_presets")
def get_llm_presets():
    """LLMサーバプリセット一覧を返す"""
    return JSONResponse(LLM_PRESETS)


@app.get("/llm_models")
def get_llm_models():
    """LLMサーバから利用可能なモデル一覧を取得する。サーバが落ちていても空配列を返す。"""
    try:
        models = client.models.list()
        return JSONResponse({"models": [m.id for m in models.data], "error": None})
    except Exception as e:
        return JSONResponse({"models": [], "error": str(e)})


@app.post("/llm_unload")
def llm_unload():
    """現在のプリセットに応じてLLMモデルをアンロードする。"""
    settings = load_settings()
    preset   = settings.get("llm_preset", "lm-studio")
    host     = (settings.get("llm_host") or "localhost").strip()
    port     = settings.get("llm_port") or 1234
    model    = (settings.get("llm_model") or "").strip()
    base     = f"http://{host}:{port}"

    def _get_json(url, timeout=5):
        req = _urllib_req.Request(url)
        with _urllib_req.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _post_json(url, body=None, timeout=10):
        data = json.dumps(body).encode() if body is not None else b""
        req = _urllib_req.Request(url, data=data,
                                   headers={"Content-Type": "application/json"},
                                   method="POST")
        with _urllib_req.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    try:
        if preset == "lm-studio":
            # ① ロード済みモデルの instance_id を取得 (/api/v0/models)
            raw = _get_json(f"{base}/api/v0/models", timeout=5)
            # {"data": [...]} 形式と直接リスト形式の両方に対応
            models_list = raw.get("data", raw) if isinstance(raw, dict) else raw
            if not isinstance(models_list, list):
                models_list = []
            loaded = [m for m in models_list if isinstance(m, dict) and m.get("state") == "loaded"]
            if not loaded:
                loaded = [m for m in models_list if isinstance(m, dict)]  # フォールバック: 全件
            if not loaded:
                return JSONResponse({"ok": False, "message": "ロード済みモデルが見つかりません"})
            # instance_id → id → path の順で探す
            instance_id = (loaded[0].get("instance_id") or loaded[0].get("id")
                           or loaded[0].get("path") or "")
            if not instance_id:
                return JSONResponse({"ok": False, "message": "モデル識別子を取得できませんでした"})
            # ② アンロード: POST /api/v1/models/unload  body: {"instance_id": "..."}
            _post_json(f"{base}/api/v1/models/unload", {"instance_id": instance_id}, timeout=10)
            return JSONResponse({"ok": True, "message": f"アンロードしました: {instance_id}"})

        elif preset == "ollama":
            if not model:
                return JSONResponse({"ok": False, "message": "モデル名が設定されていません（環境設定→モデル欄）"})
            # keep_alive: 0 でモデルをVRAMから解放
            _post_json(f"{base}/api/generate", {"model": model, "keep_alive": 0}, timeout=10)
            return JSONResponse({"ok": True, "message": f"アンロードしました: {model}"})

        elif preset == "text-gen-webui":
            _post_json(f"{base}/v1/internal/model/unload", timeout=10)
            return JSONResponse({"ok": True, "message": "モデルをアンロードしました"})

        else:
            return JSONResponse({"ok": False, "message": f"{preset} はアンロードに対応していません"})

    except Exception as e:
        return JSONResponse({"ok": False, "message": f"エラー: {e}"})


# ══════════════════════════════════════════════════════
#  キャラクター作成 / 複製 / 編集 API
#  形式: characters/<name>/ フォルダにtxt/wav/画像を格納
# ══════════════════════════════════════════════════════

from fastapi import UploadFile, File, Form
from typing import Optional

def _safe_char_name(name: str) -> str:
    """フォルダ名として安全な文字列に変換"""
    name = re.sub(r'[\\/:*?"<>|]', '_', name).strip()
    if not name:
        raise ValueError("キャラクター名が空です")
    return name

@app.post("/characters/create")
async def create_character(
    name: str = Form(...),
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
    embedding: Optional[UploadFile] = File(None),
    relationship_enabled: str = Form("false"),   # "true"=関係性の変化を有効化
):
    """新規キャラクターをサブフォルダ形式で作成する"""
    safe_name = _safe_char_name(name)
    char_dir = os.path.join(CHAR_DIR, safe_name)
    if os.path.exists(char_dir):
        raise HTTPException(400, f"キャラクター '{safe_name}' は既に存在します")
    os.makedirs(char_dir, exist_ok=True)

    # プロンプトファイル
    txt_path = os.path.join(char_dir, safe_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    # 画像ファイル
    if image and image.filename:
        ext = Path(image.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            ext = ".png"
        img_path = os.path.join(char_dir, safe_name + ext)
        with open(img_path, "wb") as f:
            f.write(await image.read())

    # 音声ファイル
    if voice and voice.filename:
        wav_path = os.path.join(char_dir, safe_name + ".wav")
        with open(wav_path, "wb") as f:
            f.write(await voice.read())

    # Speaker Embedding
    if embedding and embedding.filename:
        emb_path = os.path.join(char_dir, safe_name + ".speaker.safetensors")
        with open(emb_path, "wb") as f:
            f.write(await embedding.read())

    set_relationship_enabled(safe_name, relationship_enabled.lower() == "true")

    return JSONResponse({"status": "ok", "name": safe_name})


@app.post("/characters/{idx}/duplicate")
async def duplicate_character(
    idx: int,
    name: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
    embedding: Optional[UploadFile] = File(None),
    copy_voice: str = Form("true"),       # "false" = 複製しない（ファイルを削除）
    copy_embedding: str = Form("true"),   # "false" = 複製しない（ファイルを削除）
    relationship_enabled: str = Form("false"),   # "true"=関係性の変化を有効化
):
    """既存キャラクターを複製する。name/prompt/image/voice が指定された場合はその値を使う。"""
    state.chars = get_dynamic_characters()
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404, "キャラクターが見つかりません")

    char = state.chars[idx]
    src_dir = os.path.dirname(char["prompt"])
    base_name = _safe_char_name(char["name"].replace(" (音声なし)", ""))

    # 新しいフォルダ名を決定
    if name:
        new_name = _safe_char_name(name)
        if os.path.exists(os.path.join(CHAR_DIR, new_name)):
            raise HTTPException(400, f"キャラクター '{new_name}' は既に存在します")
    else:
        new_name = base_name + "_copy"
        counter = 2
        while os.path.exists(os.path.join(CHAR_DIR, new_name)):
            new_name = f"{base_name}_copy{counter}"
            counter += 1

    dest_dir = os.path.join(CHAR_DIR, new_name)

    # ソースフォルダをコピーしてファイル名を新名に揃える
    if os.path.isdir(src_dir) and src_dir != CHAR_DIR:
        shutil.copytree(src_dir, dest_dir)
        for fname in os.listdir(dest_dir):
            fpath = os.path.join(dest_dir, fname)
            stem = Path(fname).stem
            ext  = Path(fname).suffix
            if stem == base_name and ext in [".txt", ".wav", ".mp3"] + IMAGE_EXTS:
                os.rename(fpath, os.path.join(dest_dir, new_name + ext))
            # <base>.speaker.safetensors → <new_name>.speaker.safetensors
            elif stem == base_name + ".speaker" and ext == ".safetensors":
                os.rename(fpath, os.path.join(dest_dir, new_name + ".speaker.safetensors"))
    else:
        os.makedirs(dest_dir, exist_ok=True)
        for ext in [".txt", ".wav", ".mp3"] + IMAGE_EXTS:
            src = os.path.join(CHAR_DIR, base_name + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, new_name + ext))
        # Speaker Embedding もコピー
        src_emb = os.path.join(CHAR_DIR, base_name + ".speaker.safetensors")
        if os.path.exists(src_emb):
            shutil.copy2(src_emb, os.path.join(dest_dir, new_name + ".speaker.safetensors"))

    # プロンプトを上書き
    txt_path = os.path.join(dest_dir, new_name + ".txt")
    if prompt is not None:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    # 画像を上書き
    if image and image.filename:
        for ext in IMAGE_EXTS:
            old = os.path.join(dest_dir, new_name + ext)
            if os.path.exists(old):
                os.remove(old)
        ext = Path(image.filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            ext = ".png"
        with open(os.path.join(dest_dir, new_name + ext), "wb") as f:
            f.write(await image.read())

    # 音声を上書き（新ファイルあり）または削除（copy_voice=false）
    if voice and voice.filename:
        old_wav = os.path.join(dest_dir, new_name + ".wav")
        if os.path.exists(old_wav):
            os.remove(old_wav)
        with open(os.path.join(dest_dir, new_name + ".wav"), "wb") as f:
            f.write(await voice.read())
    elif copy_voice.lower() == "false":
        for ext in [".wav", ".mp3"]:
            p = os.path.join(dest_dir, new_name + ext)
            if os.path.exists(p):
                os.remove(p)

    # Speaker Embedding を上書き（新ファイルあり）または削除（copy_embedding=false）
    if embedding and embedding.filename:
        old_emb = os.path.join(dest_dir, new_name + ".speaker.safetensors")
        if os.path.exists(old_emb):
            os.remove(old_emb)
        with open(old_emb, "wb") as f:
            f.write(await embedding.read())
    elif copy_embedding.lower() == "false":
        p = os.path.join(dest_dir, new_name + ".speaker.safetensors")
        if os.path.exists(p):
            os.remove(p)

    # 記憶は複製しない (新しいキャラとして白紙から始める)。設定フラグのみ保存
    set_relationship_enabled(new_name, relationship_enabled.lower() == "true")

    return JSONResponse({"status": "ok", "name": new_name})


@app.get("/characters/{idx}/info")
def get_character_info(idx: int):
    """キャラクター情報（名前・プロンプト内容）を返す"""
    state.chars = get_dynamic_characters()
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404)
    char = state.chars[idx]
    prompt_text = ""
    try:
        with open(char["prompt"], "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except Exception:
        pass
    return JSONResponse({
        "name":           char["name"].replace(" (音声なし)", ""),
        "prompt":         prompt_text,
        "has_voice":      char["has_voice"],
        "has_image":      char["image_path"] is not None,
        "has_mp3":        char.get("has_mp3", False),
        "char_format":    char.get("char_format", "subfolder"),
        "voice_file":     os.path.basename(char["voice_path"]) if char.get("voice_path") else None,
        "folder":         os.path.basename(os.path.dirname(char["prompt"])),
        "has_embedding":  char.get("has_embedding", False),
        "embedding_file": os.path.basename(char["embedding_path"]) if char.get("embedding_path") else None,
        "relationship_enabled": bool(load_memory(char["name"]).get("relationship_enabled", False)),
    })


@app.post("/characters/{idx}/convert_mp3")
async def convert_char_mp3_to_wav(idx: int):
    """キャラクターの .mp3 リファレンス音声を ffmpeg で .wav に変換する（上書き）。"""
    state.chars = get_dynamic_characters()
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404, "キャラクターが見つかりません")

    char = state.chars[idx]

    if char.get("char_format") == "zip":
        raise HTTPException(400, "zip形式のキャラクターは変換できません")

    mp3_path = char.get("mp3_path")
    if not mp3_path or not os.path.exists(mp3_path):
        raise HTTPException(400, "MP3ファイルが見つかりません")

    settings   = load_settings()
    ffmpeg_cmd = get_ffmpeg_cmd(settings)
    if not ffmpeg_cmd:
        raise HTTPException(400, "ffmpegが利用できません")

    char_dir = os.path.dirname(char["prompt"])
    stem     = Path(char["prompt"]).stem
    wav_path = os.path.join(char_dir, stem + ".wav")

    try:
        subprocess.run(
            [ffmpeg_cmd, "-y", "-i", mp3_path, wav_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"ffmpeg変換失敗: {e.stderr.decode(errors='replace')}")

    state.chars = get_dynamic_characters()
    return JSONResponse({"status": "ok", "wav_path": wav_path})


@app.post("/characters/{idx}/edit")
async def edit_character(
    idx: int,
    name: str = Form(...),
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
    embedding: Optional[UploadFile] = File(None),
    relationship_enabled: str = Form("false"),   # "true"=関係性の変化を有効化
):
    """既存キャラクターの名前・プロンプト・画像・音声・Embeddingを更新する"""
    state.chars = get_dynamic_characters()
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404)

    char = state.chars[idx]
    old_name = char["name"].replace(" (音声なし)", "")
    safe_name = _safe_char_name(name)
    char_dir = os.path.dirname(char["prompt"])

    # プロンプト更新
    with open(char["prompt"], "w", encoding="utf-8") as f:
        f.write(prompt)

    # 画像更新
    if image and image.filename:
        # 旧画像を削除
        if char["image_path"] and os.path.exists(char["image_path"]):
            os.remove(char["image_path"])
        ext = Path(image.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            ext = ".png"
        img_path = os.path.join(char_dir, Path(char["prompt"]).stem + ext)
        with open(img_path, "wb") as f:
            f.write(await image.read())

    # 音声更新
    if voice and voice.filename:
        old_wav = char.get("voice_path")
        if old_wav and os.path.exists(old_wav):
            os.remove(old_wav)
        wav_path = os.path.join(char_dir, Path(char["prompt"]).stem + ".wav")
        with open(wav_path, "wb") as f:
            f.write(await voice.read())

    # Speaker Embedding 更新
    if embedding and embedding.filename:
        old_emb = char.get("embedding_path")
        if old_emb and os.path.exists(old_emb):
            os.remove(old_emb)
        emb_path = os.path.join(char_dir, Path(char["prompt"]).stem + ".speaker.safetensors")
        with open(emb_path, "wb") as f:
            f.write(await embedding.read())

    set_relationship_enabled(char["name"], relationship_enabled.lower() == "true")

    return JSONResponse({"status": "ok"})


# ══════════════════════════════════════════════════════
#  記憶 API
# ══════════════════════════════════════════════════════

def _memory_char_name(idx: int) -> str:
    state.chars = get_dynamic_characters()
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404, "キャラクターが見つかりません")
    return state.chars[idx]["name"]


def _memory_response(char_name: str) -> dict:
    mem = load_memory(char_name)
    stage, hint = relationship_stage(int(mem.get("affinity", 10)))
    key = _memory_key(char_name)
    is_selected = bool(state.selected) and _memory_key(state.selected["name"]) == key
    return {
        **mem,
        "stage":         stage,
        "stage_hint":    hint,
        "updating":      key in _memory_updating,
        "pending_turns": state.mem_user_turns if is_selected else 0,
    }


class MemoryUpdateRequest(BaseModel):
    user_facts:           Optional[list] = None
    episodes:             Optional[list] = None
    promises:             Optional[list] = None
    affinity:             Optional[int]  = None
    relationship_note:    Optional[str]  = None
    relationship_enabled: Optional[bool] = None


@app.post("/memory/flush")
def memory_flush():
    """保留中の会話を今すぐ記憶に反映する (画面を閉じる時・手動更新)"""
    started = _schedule_memory_update()
    return JSONResponse({"status": "scheduled" if started else "nothing"})


@app.get("/memory/{idx}")
def get_memory(idx: int):
    return JSONResponse(_memory_response(_memory_char_name(idx)))


@app.post("/memory/{idx}")
def post_memory(idx: int, req: MemoryUpdateRequest):
    name = _memory_char_name(idx)
    mem  = load_memory(name)
    if req.user_facts is not None:
        mem["user_facts"] = _as_str_list(req.user_facts)[-MEMORY_MAX_FACTS:]
    if req.promises is not None:
        mem["promises"] = _as_str_list(req.promises)[-MEMORY_MAX_PROMISES:]
    if req.episodes is not None:
        eps = []
        for ep in req.episodes:
            if isinstance(ep, dict) and str(ep.get("summary", "")).strip():
                eps.append({"date": str(ep.get("date", ""))[:10], "summary": str(ep["summary"]).strip()})
        mem["episodes"] = eps[-MEMORY_MAX_EPISODES:]
    if req.affinity is not None:
        mem["affinity"] = max(0, min(100, int(req.affinity)))
    if req.relationship_note is not None:
        mem["relationship_note"] = req.relationship_note.strip()[:200]
    if req.relationship_enabled is not None:
        mem["relationship_enabled"] = bool(req.relationship_enabled)
    save_memory(name, mem)
    return JSONResponse(_memory_response(name))


@app.delete("/memory/{idx}")
def delete_memory_api(idx: int):
    """記憶をリセットする (関係性の変化の有効/無効フラグは保持)"""
    name = _memory_char_name(idx)
    reset_memory(name)
    if state.selected and _memory_key(state.selected["name"]) == _memory_key(name):
        state.mem_pending    = []
        state.mem_user_turns = 0
        state.session_ctx    = {}
    return JSONResponse({"status": "ok"})


# ══════════════════════════════════════════════════════
#  会話ログ管理
# ══════════════════════════════════════════════════════

import random, string as _string

_log_session: dict = {"path": None, "char_name": None, "session_id": None}


def _make_session_id() -> str:
    return ''.join(random.choices(_string.ascii_lowercase + _string.digits, k=6))


def _log_filename(char_name: str, session_id: str) -> str:
    from datetime import datetime as _dt
    date_str  = _dt.now().strftime("%y-%m-%d")
    safe_name = re.sub(r'[\\/:*?"<>|\s]', '_', char_name)
    return f"{date_str}-{safe_name}-{session_id}.log"


def _log_header(char_name: str, session_id: str, started: str) -> str:
    return (
        f"# BridgeTTS 会話ログ\n"
        f"# Character: {char_name}\n"
        f"# Session: {session_id}\n"
        f"# Started: {started}\n"
        f"\n---\n\n"
    )


def _parse_log_file(content: str) -> dict:
    """ログファイルを解析してキャラ名とメッセージリストを返す"""
    import re as _re
    char_name = ""
    messages: list[dict] = []

    lines = content.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
        if line.startswith("# Character:"):
            char_name = line[len("# Character:"):].strip()

    body = "\n".join(lines[body_start:])
    pattern = _re.compile(
        r'\[(USER|ASSISTANT) (\d{2}:\d{2}:\d{2})\]\n(.*?)(?=\n\[(?:USER|ASSISTANT) \d{2}:\d{2}:\d{2}\]|\Z)',
        _re.DOTALL,
    )
    for m in pattern.finditer(body):
        text = m.group(3).strip()
        if text:
            messages.append({
                "role": m.group(1).lower(),   # "user" | "assistant"
                "time": m.group(2),
                "text": text,
            })

    return {"char_name": char_name, "messages": messages}


class LogStartRequest(BaseModel):
    char_name: str

class LogAppendRequest(BaseModel):
    session_id: Optional[str] = None  # 現在は無視 (グローバルセッション使用)
    role: str                          # "user" | "assistant"
    text: str

class LogRewriteRequest(BaseModel):
    messages: list  # [{role: str, text: str, time: str}]

class LogSaveRequest(BaseModel):
    session_id: Optional[str] = None  # 現在は無視
    filename:   str = ""
    directory:  str = OUTPUT_DIR

class LogLoadRequest(BaseModel):
    filepath: str


@app.post("/log/start")
def log_start(req: LogStartRequest):
    """新しい会話セッションを開始し log/ にログファイルを作成する"""
    from datetime import datetime as _dt
    os.makedirs(LOG_DIR, exist_ok=True)
    session_id = _make_session_id()
    filename   = _log_filename(req.char_name, session_id)
    path       = os.path.join(LOG_DIR, filename)
    started    = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_log_header(req.char_name, session_id, started))
    _log_session["path"]       = path
    _log_session["char_name"]  = req.char_name
    _log_session["session_id"] = session_id
    return JSONResponse({"session_id": session_id, "filename": filename})


@app.post("/log/append")
def log_append(req: LogAppendRequest):
    """現在のセッションログに1件のメッセージを追記する"""
    from datetime import datetime as _dt
    if not _log_session["path"]:
        raise HTTPException(400, "ログセッションが開始されていません")
    role_tag = "USER" if req.role == "user" else "ASSISTANT"
    time_str = _dt.now().strftime("%H:%M:%S")
    entry = f"[{role_tag} {time_str}]\n{req.text}\n\n"
    with open(_log_session["path"], "a", encoding="utf-8") as f:
        f.write(entry)
    return JSONResponse({"status": "ok"})


@app.post("/log/rewrite")
def log_rewrite(req: LogRewriteRequest):
    """現在のセッションログを messages の全内容で上書き再生成する"""
    if not _log_session["path"] or not os.path.exists(_log_session["path"]):
        raise HTTPException(400, "アクティブなセッションがありません")
    # ヘッダー部分を既存ファイルから取得 (最初の [USER/ASSISTANT] 行の手前まで)
    with open(_log_session["path"], "r", encoding="utf-8") as f:
        content = f.read()
    sep = content.find("\n[")
    header = content[:sep + 1] if sep != -1 else content
    # メッセージ部分を再生成して上書き
    with open(_log_session["path"], "w", encoding="utf-8") as f:
        f.write(header)
        for msg in req.messages:
            role_tag = "USER" if msg.get("role") == "user" else "ASSISTANT"
            time_str = msg.get("time", "??:??:??")
            f.write(f"[{role_tag} {time_str}]\n{msg['text']}\n\n")
    return JSONResponse({"status": "ok"})


@app.post("/log/save")
def log_save(req: LogSaveRequest):
    """現在のセッションログを指定ディレクトリ（デフォルト: output/）にコピー保存する"""
    if not _log_session["path"] or not os.path.exists(_log_session["path"]):
        raise HTTPException(400, "保存できるログがありません")
    save_dir = req.directory or OUTPUT_DIR
    if ".." in save_dir:
        raise HTTPException(400, "不正なパスです")
    os.makedirs(save_dir, exist_ok=True)
    filename = (req.filename or "").strip() or os.path.basename(_log_session["path"])
    if not filename.endswith(".log"):
        filename += ".log"
    dest = os.path.join(save_dir, filename)
    shutil.copy2(_log_session["path"], dest)
    return JSONResponse({"status": "ok", "filename": filename, "path": dest})


@app.post("/log/load")
def log_load(req: LogLoadRequest):
    """指定ログファイルを解析してキャラ名とメッセージリストを返す。
    filepath (絶対・相対どちらでも可) で指定する。"""
    path = req.filepath if req.filepath else None
    if not path:
        raise HTTPException(400, "filepath が指定されていません")
    if os.fspath(path).lower().endswith(".game.json"):
        raise HTTPException(400, "ゲームの閲覧用記録は会話の復元には使えません")
    if not os.path.exists(path):
        raise HTTPException(404, "ログファイルが見つかりません")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return JSONResponse(_parse_log_file(content))


@app.get("/log/files")
def log_files(directory: str = OUTPUT_DIR):
    """指定ディレクトリ（デフォルト: output/）の .log ファイル一覧を返す（更新日時降順）"""
    from datetime import datetime as _dt
    result = []
    if os.path.exists(directory):
        for fname in os.listdir(directory):
            if fname.endswith(".log"):
                fpath = os.path.join(directory, fname)
                stat  = os.stat(fpath)
                result.append({
                    "filename": fname,
                    "path":     fpath,
                    "modified": _dt.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "size":     stat.st_size,
                })
    result.sort(key=lambda x: x["modified"], reverse=True)
    return JSONResponse({"files": result})
