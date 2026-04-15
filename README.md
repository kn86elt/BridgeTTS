# BridgeTTS

LM Studio（ローカルLLM）と Irodori-TTS を組み合わせた、キャラクターチャット用 WebUI です。  
ブラウザで動作するため、PC・スマートフォン（iPhone含む）から利用できます。

---

## 必要なもの

| ソフトウェア | 用途 | 入手先 |
|---|---|---|
| **[uv](https://docs.astral.sh/uv/)** | Python パッケージ管理・実行 | https://docs.astral.sh/uv/getting-started/installation/ |
| **[LM Studio](https://lmstudio.ai/)** | ローカル LLM サーバー | https://lmstudio.ai/ |
| **[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)** | 音声合成（TTS）サーバー | GitHub 参照 |

> LM Studio と Irodori-TTS は事前に起動しておいてください。  
> LM Studio は Developer → Local Server で API モードで動作させてください。  
> Irodori-TTS は `gradio_app.py` でリファレンス音声対応サーバーとして起動してください。

---

## ファイル構成

```
bridge-webui/
├── bridge_server_api.py      # サーバー本体
├── index.html                # フロントエンド UI
├── run-bridge-webui-api.bat  # 起動スクリプト (Windows)
├── pyproject.toml            # 依存パッケージ定義
├── characters/               # キャラクターデータフォルダ
│   ├── キャラ名.txt           # キャラクター用システムプロンプト（UTF-8）
│   ├── キャラ名.wav           # 声のサンプル音声
│   └── キャラ名.png           # アイコン画像（任意）
└── prompts/                  # システムプロンプトファイル置き場（任意）
    └── 任意の名前.txt
```

### base_system_prompt.txt（任意）

`bridge_server_api.py` と同じフォルダに置くと、全キャラクター共通のシステムプロンプトとして先頭に付加されます。

### prompts/ フォルダ（任意）

サイドバーの「システムプロンプト」設定から切り替えられるプロンプトファイルを置くフォルダです。  
`.txt` ファイルを入れておくと設定画面のリストに表示されます。

---

## セットアップ

### 1. uv のインストール（初回のみ）

**Windows（PowerShell）**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 依存パッケージのインストール（初回のみ）

```bat
uv sync
```

### 3. キャラクターデータの配置

`characters/` フォルダを作成し、以下のいずれかの形式でキャラクターデータを入れてください。

#### フラット形式
```
characters/
├── きりたん.txt
├── きりたん.wav
└── きりたん.png
```

#### サブフォルダ形式
```
characters/
└── きりたん/
    ├── きりたん.txt
    ├── きりたん.wav
    └── きりたん.png
```

#### zip 形式
```
characters/
└── きりたん.zip   ← 上記どちらかの構造を zip にまとめたもの
```

---

## 起動方法

`run-bridge-webui-api.bat` をダブルクリックすると起動し、自動的にブラウザが開きます。  
開かない場合は手動で以下にアクセスしてください。

```
http://localhost:8001
```

スマートフォンからは PC の IP アドレスでアクセスできます。

```
http://<PCのIPアドレス>:8001
```

---

## チャット

- 左のサイドバーからキャラクター選択・各種設定ができます。
- 画面幅が狭い場合（スマートフォンなど）はハンバーガーメニューでサイドバーを開きます。
- キャラクターを選択すると自動で初回挨拶が生成されます。
- 空送信が可能です。何を話せばいいかわからないときに使うと、キャラクターが話を続けてくれます。
- **直接TTS**：送信バーの「直接TTS」チェックボックスをオンにすると、入力テキストを LLM に送らずそのままキャラクターのセリフとして表示・音声合成できます。

---

## 設定

### サイドバー

設定は `bridge_settings.json` に自動保存されます。

| 設定項目 | 説明 |
|---|---|
| システムプロンプト | 使用するシステムプロンプトの切り替え（後述） |
| 音声出力 | TTS の ON/OFF |
| 文章数 | 一度に TTS へ渡す文章数（0 = 全文まとめて送る） |
| ダークモード | 画面の配色を暗くする |
| 背景画像 | キャラクター画像を背景に表示する |
| ぼかし | 背景画像にぼかしエフェクトをかける |

### システムプロンプト設定

サイドバーの「⚙ プロンプト設定」ボタンから開けます。

- **ファイル (prompts/)** — `prompts/` フォルダに置いた `.txt` ファイルを選択してプレビューできます。
- **ユーザースロット（5枠）** — その場で直接テキストを書いて保存できる編集可能なスロットです。「スロット保存」で内容を保存し、OK で有効化します。
- **「キャラクタープロンプトの後ろにシステムプロンプトを読み込ませる」** — チェックをオンにするとシステムプロンプトの配置順が変わります（デフォルトはシステムプロンプト → キャラクター）。
- OK を押すと設定が保存され、会話履歴を保ったまま即座に LLM へ反映されます。

### LLM / TTS のエンドポイント変更

`run-bridge-webui-api.bat` を編集してください。

```bat
set LLM_API_URL=http://localhost:1234/v1   ← LM Studio のポート
set TTS_API_URL=http://localhost:7860/     ← Irodori-TTS のポート
```

LM Studio 以外での動作は未確認です。API キーが必要な場合は `bridge_server_api.py` を直接編集してください。

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| ブラウザが開かない | 手動で `http://localhost:8001` にアクセス |
| キャラクターが表示されない | `characters/` フォルダに `.txt` ファイルがあるか確認 |
| 音声が再生されない（iPhone） | 画面を一度タップしてから送信する（iOS のオーディオポリシーによる制限） |
| TTS サーバーに繋がらない | Irodori-TTS が起動しているか・ポート番号が合っているか確認 |
| LLM が応答しない | LM Studio でモデルが読み込まれているか確認 |
| 音声の再生が遅い | `bridge_server_api.py` の `TTS_CONFIG` を編集することで改善する可能性があります（RTX3000シリーズ以降なら `fp32` → `bf16`、`num_steps` を 10〜20 程度に下げるなど） |
