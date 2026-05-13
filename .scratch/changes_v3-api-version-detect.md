# v3 APIバージョン設定・自動検出・パラメータ振り分け

## 対象ファイル
- `bridge_server_api.py`
- `index.html`

## 変更の方針
- 環境設定モーダルに「Irodori-TTSバージョン」セレクタ（自動検出/v3/v2）を追加
- `bridge_settings.json` に `tts_api_version` として永続化（`/settings` API経由）
- 「自動検出」時は接続直後にGradio APIスキーマを調べてv2/v3を判定
- バージョンに応じて `/_run_generation` へ渡すパラメータをフィルタリング

---

## ── bridge_server_api.py ──

### 変更1: `DEFAULT_SETTINGS` に `tts_api_version` を追加

挿入位置: `"no_voice_mode": "off",` 行の直後

```python
    "tts_api_version": "auto",     # "auto"=自動検出 / "v2" / "v3"
```

---

### 変更2: `DEFAULT_TTS_CONFIG` を更新

現在の `DEFAULT_TTS_CONFIG = { ... }` ブロック全体を以下に置換:

```python
DEFAULT_TTS_CONFIG = {
    # モデル設定
    "checkpoint": "Aratako/Irodori-TTS-500M-v3",
    "model_device": "cuda",
    "model_precision": "fp32",
    "codec_device": "cuda",
    "codec_precision": "fp32",
    # 生成パラメータ (v2/v3 共通)
    "num_steps": 20,
    "num_candidates": 1,
    "seed_raw": "",
    # CFG設定 (v2/v3 共通)
    "cfg_guidance_mode": "independent",
    "cfg_scale_text": 3,
    "cfg_scale_speaker": 5,
    "cfg_scale_raw": "",
    # キャッシュ / スケーリング (v2/v3 共通)
    "context_kv_cache": True,
    "speaker_kv_scale_raw": "",
    "speaker_kv_min_t_raw": "0.9",
    "speaker_kv_max_layers_raw": "",
    # v3 専用パラメータ
    "t_schedule_mode": "sway",
    "sway_coeff": 0.0,
    # v2 専用パラメータ（v2サーバー接続時のみ送信）
    "cfg_min_t": 0.5,
    "cfg_max_t": 1,
    "truncation_factor_raw": "",
    "rescale_k_raw": "",
    "rescale_sigma_raw": "",
}
```

---

### 変更3: バージョン別キーセット定数を追加

挿入位置: `TTS_CONFIG: dict = load_tts_config()` 行の直後

```python
# バージョン別に /_run_generation へ送るパラメータキーセット
_TTS_V2_SEND_KEYS = frozenset({
    "checkpoint", "model_device", "model_precision", "codec_device", "codec_precision",
    "num_steps", "num_candidates", "seed_raw",
    "cfg_guidance_mode", "cfg_scale_text", "cfg_scale_speaker", "cfg_scale_raw",
    "cfg_min_t", "cfg_max_t",
    "context_kv_cache",
    "truncation_factor_raw", "rescale_k_raw", "rescale_sigma_raw",
    "speaker_kv_scale_raw", "speaker_kv_min_t_raw", "speaker_kv_max_layers_raw",
})
_TTS_V3_SEND_KEYS = frozenset({
    "checkpoint", "model_device", "model_precision", "codec_device", "codec_precision",
    "num_steps", "num_candidates", "seed_raw",
    "cfg_guidance_mode", "cfg_scale_text", "cfg_scale_speaker", "cfg_scale_raw",
    "t_schedule_mode", "sway_coeff",
    "context_kv_cache",
    "speaker_kv_scale_raw", "speaker_kv_min_t_raw", "speaker_kv_max_layers_raw",
})
```

---

### 変更4: `tts_client = None` の直後に `tts_api_version` グローバル変数を追加

挿入位置: `tts_client = None` 行の直後

```python
tts_api_version: str = "v2"  # get_tts_client() 接続時に設定・検出
```

---

### 変更5: `detect_tts_version()` 関数を追加

挿入位置: `def get_tts_client():` の直前

```python
def detect_tts_version(client) -> str:
    """接続中の Gradio TTS サーバーが v2 か v3 かを検出する。
    /_run_generation エンドポイントに t_schedule_mode パラメータがあれば v3。"""
    try:
        named = client._info.get("named_endpoints", {})
        run_gen = named.get("/_run_generation", [])
        param_names = [p.get("parameter_name") or p.get("label", "") for p in run_gen]
        version = "v3" if "t_schedule_mode" in param_names else "v2"
        print(f"[TTS] API version detected: {version}")
        return version
    except Exception as e:
        print(f"[TTS] Version detection failed, falling back to v2: {e}")
        return "v2"
```

---

### 変更6: `get_tts_client()` を修正

現在の `def get_tts_client():` ブロック全体を以下に置換:

```python
def get_tts_client():
    """Gradio Clientをシングルトンで取得。接続時にAPIバージョンを設定/検出する。"""
    global tts_client, tts_api_version
    if tts_client is None:
        try:
            print(f"Connecting to TTS Server at {TTS_API_URL}...")
            tts_client = Client(TTS_API_URL)
            # 設定値に応じてバージョンを決定
            configured = load_settings().get("tts_api_version", "auto")
            if configured in ("v2", "v3"):
                tts_api_version = configured
                print(f"[TTS] API version (manual): {tts_api_version}")
            else:
                tts_api_version = detect_tts_version(tts_client)
            print(f"Successfully connected to TTS Server (API: {tts_api_version}).")
        except Exception as e:
            print(f"Failed to connect to TTS Server: {e}")
    return tts_client
```

---

### 変更7: `generate_and_encode_tts()` にパラメータフィルタリングを追加

挿入位置: 関数内の `predict_kwargs = ...` の2行（現行1行）を以下に置換:

現在:
```python
        predict_kwargs = {**TTS_CONFIG, "text": text, "api_name": "/_run_generation",
                          "uploaded_audio": handle_file(voice_path) if voice_path else None}
```

変更後:
```python
        send_keys = _TTS_V3_SEND_KEYS if tts_api_version == "v3" else _TTS_V2_SEND_KEYS
        filtered_config = {k: v for k, v in TTS_CONFIG.items() if k in send_keys}
        predict_kwargs = {**filtered_config, "text": text, "api_name": "/_run_generation",
                          "uploaded_audio": handle_file(voice_path) if voice_path else None}
```

---

## ── index.html ──

### 変更8: TTS設定セクションにバージョンセレクタを追加（HTML）

挿入位置: TTS設定の `<div class="api-field-label" style="margin-top:8px">参照音声のないキャラクター</div>` の直前

```html
        <div style="margin-top:10px">
          <div class="api-field-label">Irodori-TTS バージョン</div>
          <select class="api-field-select" id="ac-tts-version" style="margin-top:4px">
            <option value="auto">自動検出</option>
            <option value="v3">v3</option>
            <option value="v2">v2</option>
          </select>
        </div>
```

---

### 変更9: DOM変数を追加

挿入位置: `const $acTtsEnvText = document.getElementById('ac-tts-env-text');` の直後

```javascript
const $acTtsVersion = document.getElementById('ac-tts-version');
```

---

### 変更10: `openApiConfigModal()` でバージョン設定を読み込む

挿入位置: 設定読み込みブロック内の `_updateTtsEnvState();` 呼び出しの直後

現在の `/settings` fetch ブロック（`$acNoVoiceOff`等を設定する箇所）に以下を追加:

追加位置: `$acNoVoiceDefault.checked = (nvm === 'default_voice');` 行の直後

```javascript
      // Irodori-TTSバージョン設定
      $acTtsVersion.value = s.tts_api_version || 'auto';
```

---

### 変更11: 適用ボタン (`$acApplyBtn`) の `/settings` POST に追加

挿入位置: `no_voice_mode: $acNoVoiceDefault.checked ? 'default_voice' : 'off',` 行の直後

```javascript
        tts_api_version: $acTtsVersion.value,
```

---

### 変更12: リセットボタン (`$acResetBtn`) にバージョンリセットを追加

挿入位置: `$acTtsPort.value = '7860';` 行の直後

```javascript
  $acTtsVersion.value = 'auto';
```
