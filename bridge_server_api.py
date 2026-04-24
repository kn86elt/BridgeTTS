"""
bridge_server.py  –  WebUI版 bridge3.py
依存: fastapi uvicorn openai python-multipart

起動:
    uvicorn bridge_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os, subprocess, time, re, threading, queue, asyncio, json, zipfile, shutil
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gradio_client import Client, handle_file
from PIL import Image
import io
import webbrowser
import base64
import requests as _requests

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

def save_settings(data: dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
DEFAULT_TTS_CONFIG = {
    # モデル設定
    "checkpoint": "Aratako/Irodori-TTS-500M-v2",
    "model_device": "cuda",
    "model_precision": "fp32",
    "codec_device": "cuda",
    "codec_precision": "fp32",
    # 生成パラメータ
    "num_steps": 20,
    "num_candidates": 1,
    "seed_raw": "",
    # CFG設定
    "cfg_guidance_mode": "independent",
    "cfg_scale_text": 3,
    "cfg_scale_speaker": 5,
    "cfg_scale_raw": "",
    "cfg_min_t": 0.5,
    "cfg_max_t": 1,
    # キャッシュ / スケーリング
    "context_kv_cache": True,
    "truncation_factor_raw": "",
    "rescale_k_raw": "",
    "rescale_sigma_raw": "",
    "speaker_kv_scale_raw": "",
    "speaker_kv_min_t_raw": "0.9",
    "speaker_kv_max_layers_raw": "",
}

def load_tts_config() -> dict:
    """tts_config.json を読み込む。なければデフォルトを書き出して返す。"""
    if os.path.exists(TTS_CONFIG_FILE):
        try:
            with open(TTS_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 新しいキーをデフォルトで補完
            merged = {**DEFAULT_TTS_CONFIG, **saved}
            return merged
        except Exception as e:
            print(f"[TTS Config] Failed to load {TTS_CONFIG_FILE}: {e}")
    # ファイルがなければデフォルトを生成
    save_tts_config(DEFAULT_TTS_CONFIG)
    return dict(DEFAULT_TTS_CONFIG)

def save_tts_config(data: dict):
    with open(TTS_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[TTS Config] Saved to {TTS_CONFIG_FILE}")

# 起動時に読み込み（グローバル変数として保持）
TTS_CONFIG: dict = load_tts_config()


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


# --- 共通クライアントの保持 ---
tts_client = None

def get_tts_client():
    """Gradio Clientをシングルトンで取得"""
    global tts_client
    if tts_client is None:
        try:
            print(f"Connecting to TTS Server at {TTS_API_URL}...")
            tts_client = Client(TTS_API_URL)
            print("Successfully connected to TTS Server.")
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

        text, voice_path, req_id = item
        try:
            audio_b64 = generate_and_encode_tts(text, voice_path)
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

    return {
        "name":       base if has_voice else f"{base} (音声なし)",
        "prompt":     txt_path,
        "voice_path": active_voice,
        "mp3_path":   mp3_path if has_mp3 else None,
        "image_path": _find_image(directory, base),
        "has_voice":  has_voice,
        "has_mp3":    has_mp3,
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

state = AppState()

# --- FastAPI ---
app = FastAPI(title="Bridge WebUI")

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
        "index": i, 
        "name": c["name"], 
        "has_image": c["image_path"] is not None,
        "has_voice": c["has_voice"]
    } for i, c in enumerate(state.chars)]

# --- キャラクター画像 ---
@app.get("/characters/{idx}/image")
def get_character_image(idx: int):
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404)

    img_path = state.chars[idx].get("image_path")
    if not img_path or not os.path.exists(img_path):
        raise HTTPException(404)

    img = Image.open(img_path).convert("RGB")
    img = crop_center(img)
    img = img.resize((640, 640), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")

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


# --- キャラクター選択 ---
class SelectRequest(BaseModel):
    index: int

@app.post("/select")
def select_character(req: SelectRequest):
    state.chars = get_dynamic_characters()
    if req.index < 0 or req.index >= len(state.chars):
        raise HTTPException(400, "Invalid index")

    state.selected = state.chars[req.index]

    base_prompt = ""
    if os.path.exists(BASE_PROMPT_FILE):
        with open(BASE_PROMPT_FILE, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()

    with open(state.selected["prompt"], "r", encoding="utf-8") as f:
        char_setting = f.read().strip()

    # システムプロンプトの合成
    settings = load_settings()
    sys_prompt_text = _build_system_prompt(base_prompt, char_setting, settings)

    state.memory = [{"role": "system", "content": sys_prompt_text}]
    return {"status": "ok", "name": state.selected["name"]}


@app.post("/update_system_prompt")
def update_system_prompt():
    """システムプロンプトを再構築し、会話履歴をリセットする。
    LLMへ渡す履歴は system メッセージのみになるため文脈の汚染を防ぐ。
    UI側のログ表示はフロントエンドが保持するため見た目は変化しない。"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")

    with open(state.selected["prompt"], "r", encoding="utf-8") as f:
        char_setting = f.read().strip()

    settings = load_settings()
    sys_prompt_text = _build_system_prompt("", char_setting, settings)

    # 会話履歴を完全リセット（system メッセージのみ残す）
    state.memory = [{"role": "system", "content": sys_prompt_text}]

    return {"status": "ok"}


@app.get("/status")
async def get_status():
    is_processing = not generate_queue.empty()
    return {"is_generating": is_processing}


# --- チャット ---
class ChatRequest(BaseModel):
    message: str
    tts_enabled: bool = True


# --- 補助関数 ---
def generate_and_encode_tts(text, voice_path):
    """TTSを生成してBase64文字列で返す。voice_path=None の場合はデフォルト音声で生成。"""
    tts = get_tts_client()
    if tts is None:
        return None

    try:
        predict_kwargs = {**TTS_CONFIG, "text": text, "api_name": "/_run_generation",
                          "uploaded_audio": handle_file(voice_path) if voice_path else None}
        result = tts.predict(**predict_kwargs)
        
        res_data = result[0]
        gen_path = res_data["value"] if isinstance(res_data, dict) and "value" in res_data else res_data
        
        if gen_path and os.path.exists(gen_path):
            with open(gen_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"TTS Error: {e}")
    return None


def enqueue_tts(sentence: str, voice_path: str, req_id: str):
    """キューに積む前に queued カウンターをインクリメントする（必ずペアで使う）"""
    with finished_audios_lock:
        if req_id in finished_audios:
            finished_audios[req_id]["queued"] += 1
    generate_queue.put((sentence, voice_path, req_id))


EMOJI_PAT = re.compile(
    r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FFFF\U00002300-\U000023FF]+'
)

def clean_for_tts(text: str) -> str:
    """括弧内を除去するのみ。絵文字は元の位置のまま残す。
    irodori-TTSは文末の絵文字に反応するため、位置を変えずそのまま渡す。"""
    s = re.sub(r'[\(（].*?[\)）]', '', text)
    return s.strip()


def stream_llm_with_tts(messages: list, tts_enabled: bool):
    req_id = f"req_{int(time.time() * 1000)}"

    with finished_audios_lock:
        finished_audios[req_id] = {"audios": [], "queued": 0, "completed": 0}

    settings = load_settings()
    tts_batch_size: int = settings.get("tts_batch_size", 0)  # 0=すべて
    no_voice_mode: str  = settings.get("no_voice_mode", "off")

    # TTS使用可否: 音声ありキャラ、またはデフォルト音声モードで音声なしキャラ
    tts_available = (
        state.selected.get("has_voice")
        or no_voice_mode == "default_voice"
    )
    # デフォルト音声モード時は voice_path=None でTTSを呼ぶ
    tts_voice_path = state.selected.get("voice_path") if state.selected.get("has_voice") else None

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
                enqueue_tts(s, tts_voice_path, req_id)
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
                        enqueue_tts(s, tts_voice_path, req_id)
                else:
                    # 残りをバッチサイズで処理（端数は1つにまとめる）
                    while len(pending_sentences) >= tts_batch_size:
                        chunk = pending_sentences[:tts_batch_size]
                        pending_sentences = pending_sentences[tts_batch_size:]
                        combined = "".join(chunk)
                        s = clean_for_tts(combined)
                        if len(s) > 1:
                            enqueue_tts(s, tts_voice_path, req_id)
                    # 端数
                    if pending_sentences:
                        combined = "".join(pending_sentences)
                        pending_sentences = []
                        s = clean_for_tts(combined)
                        if len(s) > 1:
                            enqueue_tts(s, tts_voice_path, req_id)

        state.memory.append({"role": "assistant", "content": full_answer})

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
            enqueue_tts(friendly, tts_voice_path, req_id)
            yield f"data: {json.dumps({'done': True, 'req_id': req_id, 'error_detail': err_str})}\n\n"
            # finished_audios の解放はポーリング側（audio_poll）に任せる
        else:
            yield f"data: {json.dumps({'done': True, 'req_id': None, 'error_detail': err_str})}\n\n"
            with finished_audios_lock:
                finished_audios.pop(req_id, None)


@app.post("/chat")
def chat(req: ChatRequest):
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")

    user_text = req.message if req.message.strip() else " "
    state.memory.append({"role": "user", "content": user_text})

    return StreamingResponse(
        stream_llm_with_tts(state.memory, req.tts_enabled),
        media_type="text/event-stream"
    )


@app.get("/initial_greeting")
def initial_greeting(tts_enabled: bool = True):
    if not state.selected:
        raise HTTPException(400, "Character not selected")
        
    prompt = "会話開始時の最初の一言として、短く自然な挨拶をしてください。"
    state.memory.append({"role": "user", "content": prompt})

    return StreamingResponse(
        stream_llm_with_tts(state.memory, tts_enabled),
        media_type="text/event-stream"
    )


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


# --- 直接TTS (LLMを通さずTTSだけ実行) ---
class DirectTTSRequest(BaseModel):
    text: str

@app.post("/tts_direct")
def tts_direct(req: DirectTTSRequest):
    """入力テキストをそのままTTSに渡して音声を返す"""
    if not state.selected:
        raise HTTPException(400, "キャラクターが選択されていません")
    voice_path = state.selected.get("voice_path")
    settings = load_settings()
    no_voice_mode = settings.get("no_voice_mode", "off")
    if not voice_path and no_voice_mode != "default_voice":
        raise HTTPException(400, "音声ファイルが設定されていません")

    text = clean_for_tts(req.text.strip())
    if not text:
        raise HTTPException(400, "テキストが空です")

    audio_b64 = generate_and_encode_tts(text, voice_path)
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
    """設定を保存する"""
    current = load_settings()
    current.update(data)
    save_settings(current)
    return JSONResponse({"status": "ok"})


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
def get_tts_config_api():
    """TTS詳細設定を返す"""
    return JSONResponse(load_tts_config())

@app.post("/tts_config")
def post_tts_config_api(data: dict = Body(...)):
    """TTS詳細設定を保存する"""
    current = load_tts_config()
    current.update(data)
    save_tts_config(current)
    # グローバル変数も更新
    global TTS_CONFIG
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
    settings = load_settings()
    no_voice_mode = settings.get("no_voice_mode", "off")
    if not voice_path and no_voice_mode != "default_voice":
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
            enqueue_tts(s, voice_path, req_id)

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
    settings = load_settings()
    no_voice_mode = settings.get("no_voice_mode", "off")
    if not voice_path and no_voice_mode != "default_voice":
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
        audio_b64 = generate_and_encode_tts(s, voice_path)
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

    try:
        if preset == "lm-studio":
            # ① ロード済みモデルの identifier を取得
            r = _requests.get(f"{base}/api/v0/models", timeout=5)
            r.raise_for_status()
            models_list = r.json()
            loaded = [m for m in models_list if m.get("state") == "loaded"]
            if not loaded:
                loaded = models_list  # フォールバック: 全件
            if not loaded:
                return JSONResponse({"ok": False, "message": "ロード済みモデルが見つかりません"})
            identifier = loaded[0].get("id") or loaded[0].get("path") or ""
            if not identifier:
                return JSONResponse({"ok": False, "message": "モデル識別子を取得できませんでした"})
            # ② アンロード
            r2 = _requests.post(f"{base}/api/v0/models/unload",
                                 json={"identifier": identifier}, timeout=10)
            r2.raise_for_status()
            return JSONResponse({"ok": True, "message": f"アンロードしました: {identifier}"})

        elif preset == "ollama":
            if not model:
                return JSONResponse({"ok": False, "message": "モデル名が設定されていません（環境設定→モデル欄）"})
            # keep_alive: 0 でモデルをVRAMから解放
            r = _requests.post(f"{base}/api/generate",
                                json={"model": model, "keep_alive": 0}, timeout=10)
            r.raise_for_status()
            return JSONResponse({"ok": True, "message": f"アンロードしました: {model}"})

        elif preset == "text-gen-webui":
            r = _requests.post(f"{base}/v1/internal/model/unload", timeout=10)
            r.raise_for_status()
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

    return JSONResponse({"status": "ok", "name": safe_name})


@app.post("/characters/{idx}/duplicate")
async def duplicate_character(
    idx: int,
    name: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
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
    else:
        os.makedirs(dest_dir, exist_ok=True)
        for ext in [".txt", ".wav", ".mp3"] + IMAGE_EXTS:
            src = os.path.join(CHAR_DIR, base_name + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, new_name + ext))

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

    # 音声を上書き
    if voice and voice.filename:
        old_wav = os.path.join(dest_dir, new_name + ".wav")
        if os.path.exists(old_wav):
            os.remove(old_wav)
        with open(os.path.join(dest_dir, new_name + ".wav"), "wb") as f:
            f.write(await voice.read())

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
        "name":        char["name"].replace(" (音声なし)", ""),
        "prompt":      prompt_text,
        "has_voice":   char["has_voice"],
        "has_image":   char["image_path"] is not None,
        "has_mp3":     char.get("has_mp3", False),
        "char_format": char.get("char_format", "subfolder"),
        "voice_file":  os.path.basename(char["voice_path"]) if char.get("voice_path") else None,
        "folder":      os.path.basename(os.path.dirname(char["prompt"])),
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
):
    """既存キャラクターの名前・プロンプト・画像・音声を更新する"""
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
