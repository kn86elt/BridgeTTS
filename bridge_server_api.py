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

# --- 設定ファイル ---
SETTINGS_FILE = "bridge_settings.json"
TTS_CONFIG_FILE = "tts_config.json"   # TTS詳細設定の外部ファイル

PROMPTS_DIR = "prompts"

DEFAULT_SETTINGS = {
    "tts_batch_size": 2,      # TTSに投げる文章数 (0=すべて)
    "dark_mode": False,
    "blur_mode": True,
    "tts_enabled": True,
    "bg_enabled": True,
    # システムプロンプト設定
    "system_prompt_file": "__base__",   # 初回デフォルト = base_system_prompt.txt
    "system_prompt_after_char": False,  # True=キャラクタープロンプトの後ろにシステムプロンプトを配置
    "user_slots": ["", "", "", "", ""],  # ユーザー編集スロット x5
    "active_user_slot": -1,             # 使用中のユーザースロット (-1=なし)
    "slot_and_file": False,             # True=スロットとファイルを同時適用
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
# LM StudioのAPIモード設定
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1")
LLM_API_KEY = "lm-studio"


# irodori.TTSの設定
TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:7862/")

# その他の設定
CHAR_DIR        = "characters"
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
                zf.extractall(dest)
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
    has_voice = os.path.exists(wav_path)
    img_path  = _find_image(directory, base)

    # wav と同名でなくとも .wav が1つだけあれば使う
    if not has_voice:
        wavs = [f for f in os.listdir(directory) if f.endswith(".wav")]
        if wavs:
            wav_path  = os.path.join(directory, wavs[0])
            has_voice = True

    return {
        "name":       base if has_voice else f"{base} (音声なし)",
        "prompt":     txt_path,
        "voice_path": wav_path if has_voice else None,
        "image_path": img_path,
        "has_voice":  has_voice,
    }


def get_dynamic_characters() -> list:
    if not os.path.exists(CHAR_DIR):
        os.makedirs(CHAR_DIR)
        return []

    char_list: list[dict] = []
    seen_names: set[str]  = set()   # 重複防止

    entries = os.listdir(CHAR_DIR)

    for entry in sorted(entries):
        full_path = os.path.join(CHAR_DIR, entry)

        # ── パターン1: フラット (.txt) ──────────────────────────
        if entry.endswith(".txt"):
            base = Path(entry).stem
            if base in seen_names:
                continue
            char = _char_from_dir(base, CHAR_DIR)
            if char:
                seen_names.add(base)
                char_list.append(char)

        # ── パターン2: サブフォルダ ──────────────────────────────
        elif os.path.isdir(full_path) and entry != ".zip_cache":
            base = entry
            if base in seen_names:
                continue
            char = _char_from_dir(base, full_path)
            if char:
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
                seen_names.add(base)
                char_list.append(char)

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
    """system_prompt_file の値からテキストを読んで返す。
    '__base__' は base_system_prompt.txt を指す特別値。"""
    if not fname:
        return ""
    if fname == "__base__":
        fpath = BASE_PROMPT_FILE
    else:
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
    （__base__ 選択時は _read_file_prompt 内で処理される）。"""
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
    """TTSを生成してBase64文字列で返す"""
    tts = get_tts_client()
    if tts is None:
        return None
    
    try:
        result = tts.predict(
            **TTS_CONFIG,
            text=text,
            uploaded_audio=handle_file(voice_path),
            api_name="/_run_generation"
        )
        
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
    tts_batch_size: int = settings.get("tts_batch_size", 2)  # 0=すべて

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
                enqueue_tts(s, state.selected["voice_path"], req_id)
                enqueued_count += 1

    try:
        stream = client.chat.completions.create(
            model="local-model",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_answer += delta
            llm_buffer += delta

            yield f"data: {json.dumps({'token': delta})}\n\n"

            if tts_enabled and state.selected.get("has_voice"):
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
        if tts_enabled and state.selected.get("has_voice"):
            if llm_buffer.strip():
                pending_sentences.append(llm_buffer)
            if pending_sentences:
                if tts_batch_size == 0:
                    # 0=すべてまとめて → 全文を1リクエストで送る
                    combined = "".join(pending_sentences)
                    pending_sentences = []
                    s = clean_for_tts(combined)
                    if len(s) > 1:
                        enqueue_tts(s, state.selected["voice_path"], req_id)
                else:
                    # 残りをバッチサイズで処理（端数は1つにまとめる）
                    while len(pending_sentences) >= tts_batch_size:
                        chunk = pending_sentences[:tts_batch_size]
                        pending_sentences = pending_sentences[tts_batch_size:]
                        combined = "".join(chunk)
                        s = clean_for_tts(combined)
                        if len(s) > 1:
                            enqueue_tts(s, state.selected["voice_path"], req_id)
                    # 端数
                    if pending_sentences:
                        combined = "".join(pending_sentences)
                        pending_sentences = []
                        s = clean_for_tts(combined)
                        if len(s) > 1:
                            enqueue_tts(s, state.selected["voice_path"], req_id)

        state.memory.append({"role": "assistant", "content": full_answer})

        # LLM完了を即通知。残音声があれば req_id をフロントに渡してポーリングさせる
        with finished_audios_lock:
            q = finished_audios[req_id]["queued"]
        has_pending = tts_enabled and state.selected.get("has_voice") and q > 0
        yield f"data: {json.dumps({'done': True, 'req_id': req_id if has_pending else None})}\n\n"

        # req_id を返した場合はポーリング側が解放するので、ここでは pop しない
        if not has_pending:
            with finished_audios_lock:
                finished_audios.pop(req_id, None)

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
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
    if not voice_path:
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
    """プロンプトファイル一覧を返す。
    先頭に __base__ (base_system_prompt.txt) を含む。"""
    result = []
    # base_system_prompt.txt を先頭に追加
    if os.path.exists(BASE_PROMPT_FILE):
        result.append("__base__")
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    files = sorted([
        f for f in os.listdir(PROMPTS_DIR)
        if f.endswith(".txt") and os.path.isfile(os.path.join(PROMPTS_DIR, f))
    ])
    result.extend(files)
    return JSONResponse(result)


@app.get("/prompts/{filename}")
def get_prompt(filename: str):
    """プロンプトファイルの内容を返す。__base__ は base_system_prompt.txt を参照。"""
    if filename == "__base__":
        if not os.path.exists(BASE_PROMPT_FILE):
            raise HTTPException(404, "base_system_prompt.txt not found")
        with open(BASE_PROMPT_FILE, "r", encoding="utf-8") as f:
            return JSONResponse({"filename": "__base__", "content": f.read()})
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
def duplicate_character(idx: int):
    """既存キャラクターを複製する"""
    state.chars = get_dynamic_characters()
    if idx < 0 or idx >= len(state.chars):
        raise HTTPException(404, "キャラクターが見つかりません")

    char = state.chars[idx]
    src_dir = os.path.dirname(char["prompt"])

    # 新しいフォルダ名を決定 (name_copy, name_copy2, ...)
    base_name = _safe_char_name(char["name"].replace(" (音声なし)", ""))
    new_name = base_name + "_copy"
    counter = 2
    while os.path.exists(os.path.join(CHAR_DIR, new_name)):
        new_name = f"{base_name}_copy{counter}"
        counter += 1

    dest_dir = os.path.join(CHAR_DIR, new_name)

    # フォルダをコピー
    if os.path.isdir(src_dir) and src_dir != CHAR_DIR:
        shutil.copytree(src_dir, dest_dir)
        # ファイル名をフォルダ名に合わせてリネーム
        for fname in os.listdir(dest_dir):
            fpath = os.path.join(dest_dir, fname)
            stem = Path(fname).stem
            ext  = Path(fname).suffix
            # 古いステム名と一致するファイルをリネーム
            if stem == base_name and ext in [".txt", ".wav"] + IMAGE_EXTS:
                os.rename(fpath, os.path.join(dest_dir, new_name + ext))
    else:
        # フラットファイルをコピーしてフォルダに移動
        os.makedirs(dest_dir, exist_ok=True)
        for ext in [".txt", ".wav"] + IMAGE_EXTS:
            src = os.path.join(CHAR_DIR, base_name + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, new_name + ext))

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
        "name": char["name"].replace(" (音声なし)", ""),
        "prompt": prompt_text,
        "has_voice": char["has_voice"],
        "has_image": char["image_path"] is not None,
        "folder": os.path.basename(os.path.dirname(char["prompt"])),
    })


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
