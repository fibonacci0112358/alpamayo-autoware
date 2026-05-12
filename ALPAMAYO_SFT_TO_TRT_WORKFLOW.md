# Alpamayo 1.5 SFT → TRT エクスポート ワークフロー

このドキュメントは、Alpamayo 1.5 をファインチューニング（SFT）し、TensorRT エンジンにエクスポートするまでの完全な手順です。

**全体フロー:**

```
①Base Checkpoint (Alpamayo1.5)
    ↓
②Stage1 SFT (VLM LoRA)  →  LoRA adapter (Stage1)
    ↓
③Merge LoRA → merged_stage1
    ↓
④Stage2 SFT: 2つのパターン
    ├─ パターン A: LoRA 使用
    │   ↓
    │   ④-A1 SFT → LoRA adapter (Stage2)
    │   ↓
    │   ⑤ Merge LoRA → merged_final
    │   ↓
    │   ⑥ ONNX export + INT8 quantize + TRT build
    │
    └─ パターン B: 全パラメータ SFT
        ↓
        ④-B SFT → checkpoint_stage2
        ↓
        ⑥ ONNX export + INT8 quantize + TRT build
    ↓
⑦TRT engine ready for inference
```

**要点:**
- **Stage1:** VLM を LoRA で調整（推奨）
- **Stage2:** LoRA または全パラメータ SFT を選択
  - LoRA: メモリ効率的、ステップ 5 でマージ必要
  - 全パラメータ SFT: より強力な調整、マージ不要
- **venv の役割:**
  - `a1_5_venv` = Alpamayo1.5 の公式 runtime で使用する venv
  - `.venv-trt` = TensorRT エンジン生成専用の build venv

---

## 前提条件

- **GPU環境:** CUDA 12.x (または 11.x、ご自身の環境に合わせて)
- **Python バージョン:**
  - 学習 & TRT build: Python 3.12 (推奨)
  - ROS 2 Humble runtime: Python 3.10 (推奨)
- **ストレージ:** 50GB以上 (モデル、チェックポイント、キャッシュ用)
- **メモリ:** GPU 24GB 以上推奨 (小さい場合は `batch_size` / `num_calibration_samples` を下げる)

---

## ステップ 1: 環境構築

### 1.1 Python 3.12 venv を作成（学習 & TRT build 用）

```bash
cd /home/yutotakeuchi/alpamayo-autoware
source a1_5_venv/bin/activate
pip install --upgrade pip
```

### 1.2 学習用依存をインストール

学習とTRT build の依存をインストール:

```bash
pip install -r scripts/requirements-trt-build.txt
pip install peft==0.14.0  # LoRA support
```

エラーが出た場合は `ALPAMAYO_FINETUNE_TRT_INTEGRATION.md` のトラブルシュートを参照してください。

---

## ステップ 2: Stage1 — VLM SFT (LoRA)

### 2.1 データ準備

学習用データセットを準備してください。現在の実装では、Stage1 / Stage2 ともに
`.pt` サンプルを manifest で指す形が最も扱いやすいです。

最低限の manifest 形式:

- `file`: `.pt` サンプルへの相対パスまたは絶対パス
- `clip_id`: サンプル識別子
- `t0_us`: keyframe のタイムスタンプ

`.pt` サンプルに含める代表的なキー:

- `image_frames`: `[num_cameras, num_frames_per_camera, C, H, W]` または `[N, C, H, W]`
- `camera_indices`: `[num_cameras]`
- `ego_history_xyz`: `[1, 16, 3]`
- `ego_history_rot`: `[1, 16, 3, 3]`
- `ego_future_xyz`: `[1, 64, 3]`
- `ego_future_rot`: `[1, 64, 3, 3]`
- `relative_timestamps`, `absolute_timestamps`, `clip_id`, `t0_us`

Stage1 では主に `image_frames` と `camera_indices`、Stage2 ではさらに
`ego_history_*` / `ego_future_*` が必要です。`PtManifestDataset` はこれらを
そのまま読み、`PtManifestCollator` が chat template 経由で `input_ids` と
`labels` を作ります。

詳細は `finetune/sft/data/pt_manifest_dataset.py` と
`finetune/sft/data/pai_av_dataset.py` を参照してください。

### 2.2 Stage1 小バッチテスト（確認用）

```bash
cd /home/yutotakeuchi/alpamayo-autoware
source a1_5_venv/bin/activate

python3 finetune/sft/tests/run_small_batch.py \
  --use-vlm \
  --vlm-name-or-path /path/to/base_checkpoint_or_hf_id \
  --device cuda \
  --batch-size 2 \
  --num-steps 5
```

出力: loss が減少していれば OK。

### 2.3 本学習 (LoRA adapter 生成)

トレーニングスクリプト、またはお手持ちの HuggingFace Trainer を使用してください:

**パターン A: 外部ローカルスクリプト (e.g., alpamayo repo から)**
```bash
python3 /path/to/your/training_script.py \
  --model-id /path/to/base_checkpoint \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.1 \
  --output-dir ./lora_stage1 \
  --dataset-path /path/to/your/dataset
```

**パターン B: 本リポジトリ内の雛形 + Trainer**
```bash
python3 finetune/sft/train_hf_1_5.py \
  model.vlm_name_or_path=/path/to/base_checkpoint \
  model.freeze_vlm=false \
  lora.r=32 \
  lora.alpha=64 \
  output_dir=./lora_stage1
```

**出力物:**
- `./lora_stage1/adapter_config.json` + `adapter_model.bin`
- チェックポイント (オプション)

---

## ステップ 3: LoRA マージ & 検証 (Stage1)

### 3.1 LoRA アダプターをマージ

```bash
python3 finetune/sft/scripts/merge_lora_and_save.py \
  --base-checkpoint /path/to/base_checkpoint \
  --lora-dir ./lora_stage1 \
  --output-dir ./merged_stage1 \
  --tokenizer-dir /path/to/tokenizer  # (optional, if different from base)
```

**出力物:** `./merged_stage1/` — マージ済 checkpoint

### 3.2 マージ後の検証

```bash
python3 finetune/sft/scripts/verify_merged_checkpoint.py \
  --merged-dir ./merged_stage1
```

期待出力:
```
Successfully loaded Alpamayo1_5 from ./merged_stage1
Model type: Alpamayo1_5
Num parameters: XXXXXXXXXX
```

エラーが出た場合は `tokenizer` / `config.json` のコピーを確認してください。

---

## ステップ 4: Stage2 — Action/Expert SFT

このリポジトリには `finetune/sft/train_hf_1_5_stage2.py` と
`finetune/sft/models/sft_alpamayo_1_5_stage2.py` があります。Stage2 は
action/expert diffusion loss を持っていて、まずは「モデルが読めるか」と
「最小の forward/loss 経路が通るか」を確認するのが実用的です。

### 4.1 データ準備 (Stage2)

Stage2 の学習データを準備:

- `manifest.json` で `.pt` サンプルを列挙
- 各 `.pt` に `image_frames` と `ego_history_*` / `ego_future_*` を保存
- 画像は `image_root` から読めるように相対パスで持つのが楽
- `chunk_ids` または `file_start` / `file_end` で train / valid を分ける

Stage2 の collator は `input_ids` を作った上で `labels` を自動生成します。
したがって、学習前に確認すべきなのは「manifest が読めるか」「`.pt`
から trajectory tensor が揃うか」「`image_frames` の shape が揃っているか」です。

学習前チェックの最小条件:

- `python3 - <<'PY'` で `json.load(manifest)` が通る
- 先頭 `.pt` に `image_frames` / `ego_history_xyz` / `ego_future_xyz` がある
- `image_frames` の shape が `torch.uint8` の 4D/5D tensor である
- `ego_history_xyz` が `[1, 16, 3]`、`ego_future_xyz` が `[1, 64, 3]` である

確認用コマンド:

```bash
PYTHONPATH=$PWD:$PWD/src python3 finetune/sft/scripts/inspect_stage2_manifest.py \
  --manifest 206_dataset_ver1/manifest.json \
  --image-root 206_dataset_ver1 \
  --max-samples 1
```

詳細は `finetune/sft/data/pt_manifest_dataset.py` と `finetune/sft/data/pai_av_dataset.py` を参照。

### 4.2 Stage2 スモークテスト

まず、import と最小ロジックの確認をします。前提として `PYTHONPATH=$PWD:$PWD/src`
を付け、依存の `einops` も入れてください。

```bash
cd /home/yutotakeuchi/alpamayo-autoware
source a1_5_venv/bin/activate
PYTHONPATH=$PWD:$PWD/src python3 finetune/sft/tests/run_small_batch.py --device cpu --stage2
```

確認ポイント:

- toy model の forward/backward が通る
- `TrainableAlpamayo1_5_Stage2` の import が通る
- `compute_action_loss` の static テストまで到達する

このテストは、実データを入れる前に Stage2 の loss 計算経路が通るかを確認するためのものです。

### 4.3 本学習 (Stage2)

実際の action/expert SFT を回す場合は、Stage1 で作った merged checkpoint を使い、
`train_hf_1_5_stage2.py` を起点にします。

```bash
cd /home/yutotakeuchi/alpamayo-autoware
source a1_5_venv/bin/activate
PYTHONPATH=$PWD:$PWD/src python3 finetune/sft/train_hf_1_5_stage2.py \
  model.config.vlm_name_or_path=Alpamayo-1.5-10B \
  model.stage1_vlm_checkpoint_path=outputs/stage1_full_merged \
  lora.use_lora=true \
  training.output_dir=outputs/stage2_lora
```

このコマンドは `finetune/sft/configs/stage2.yaml` の既定値に従って、
`206_dataset_ver1/manifest.json` と `206_dataset_ver1/` をそのまま使います。
分割は `train_file_start=0, train_file_end=60, valid_file_start=60,
valid_file_end=76` です。

**注意:** Stage2 は loss 実装済みですが、batch 形状や `config.vlm_name_or_path`
の指定が崩れると zero-loss フォールバックに落ちます。実学習前に merged stage1
checkpoint でスモークテストを通してください。

---

## ステップ 5: LoRA マージ & 検証 (Stage2) [LoRA パターンのみ]

**注:** 全パラメータ SFT を選んだ場合はこのステップをスキップして、ステップ 6 へ進んでください。

### 5.1 Stage2 LoRA をマージ

```bash
python3 finetune/sft/scripts/merge_lora_and_save.py \
  --base-checkpoint ./merged_stage1 \
  --lora-dir ./lora_stage2 \
  --output-dir ./merged_final \
  --tokenizer-dir /path/to/tokenizer
```

**出力物:** `./merged_final/` — Stage1 + Stage2 マージ済 checkpoint

### 5.2 最終検証

```bash
python3 finetune/sft/scripts/verify_merged_checkpoint.py \
  --merged-dir ./merged_final
```

期待: 正常にロード。

---

## ステップ 6: ONNX エクスポート + INT8 量子化 + TRT ビルド

### 6.1 TRT build 用の venv を確認

すでに `.venv-trt` がある場合はそれを使用。ない場合は作成:

```bash
python3.12 -m venv .venv-trt
source .venv-trt/bin/activate
pip install -r scripts/requirements-trt-build.txt
```

### 6.2 TRT エンジン生成

**注:** LoRA パターンの場合は `./merged_final` を、全パラメータ SFT の場合は `./checkpoint_stage2` を指定してください。

```bash
source .venv-trt/bin/activate

# LoRA パターン
python3 scripts/build_trt_expert_engine.py \
  --model-id ./merged_final \
  --output-dir ./engines_trt \
  --num-calibration-samples 8 \
  --calibration-method entropy \
  --skip-validation false

# または全パラメータ SFT パターン
python3 scripts/build_trt_expert_engine.py \
  --model-id ./checkpoint_stage2 \
  --output-dir ./engines_trt \
  --num-calibration-samples 8 \
  --calibration-method entropy \
  --skip-validation false
```

**実行内容:**
1. 推論を実行し、expert denoiser 入力をキャプチャ
2. ONNX にエクスポート (float32)
3. SmoothQuant (INT8) で量子化
4. TensorRT エンジン生成

**実行時間:** 15–30 分（GPU, calibration set サイズに依存）

**出力物:**
```
./engines_trt/
  ├── expert.fp32.onnx
  ├── expert.int8.onnx
  └── expert.trt
```

### 6.3 トラブルシュート

**症状: OOM during calibration**
```bash
python3 scripts/build_trt_expert_engine.py \
  --model-id ./merged_final \
  --output-dir ./engines_trt \
  --num-calibration-samples 4 \
  --calibration-method minmax \
  --skip-validation
```

**症状: ONNX export error**
→ スクリプトは自動で dtype を float32 に変換しますが、custom forward があれば確認してください。

詳細は `ALPAMAYO_FINETUNE_TRT_INTEGRATION.md` を参照。

---

## ステップ 7: 推論で TRT エンジンを使用

### 7.1 推論ノード起動

```bash
source a1_5_venv/bin/activate  # Alpamayo1.5 の公式 runtime venv (ROS 2 Python 3.10)

ros2 launch alpamayo_ros alpamayo.launch.py \
  expert_engine_path:=./engines_trt/expert.trt
```

または、Python スクリプトで直接:

```python
from src.alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

# LoRA パターン（マージ済）
model = Alpamayo1_5.from_pretrained("./merged_final")

# または全パラメータ SFT パターン
model = Alpamayo1_5.from_pretrained("./checkpoint_stage2")

# Inference runs using TRT engine if provided
outputs = model.sample_trajectories_from_data_with_vlm_rollout(...)
```

### 7.2 検証

推論結果が期待通りか確認:

```bash
python3 visualize_trajectory.py \
  --input-image input_images/scene_001.jpg \
  --output outputs/
```

---

## よくある質問

### Q1: Stage1 だけをやりたい (Stage2 スキップ)

```bash
# Stage1 をマージ
python3 finetune/sft/scripts/merge_lora_and_save.py \
  --base-checkpoint /path/to/base \
  --lora-dir ./lora_stage1 \
  --output-dir ./merged_stage1

# 直接 TRT ビルド
python3 scripts/build_trt_expert_engine.py \
  --model-id ./merged_stage1 \
  --output-dir ./engines_trt
```

### Q2: Stage2 で LoRA vs 全パラメータ SFT、どちらを使うべき？

| 方式 | メリット | デメリット | 推奨用途 |
|------|---------|----------|--------|
| **LoRA** | メモリ削減、高速学習、推論速度変わらず | 微調整表現力に制限 | メモリ制約、迅速な実験 |
| **全パラメータ SFT** | より強力な調整、Alpamayo1 同じアプローチ | GPU メモリ多く必要、学習時間増加 | メモリ充分、品質重視 |

**判断基準:**
- **メモリ < 24GB** → LoRA を使用（ステップ 4.3 パターン A）
- **メモリ >= 24GB** → 全パラメータ SFT を使用（ステップ 4.3 パターン B）

全パラメータ SFT を選んだ場合、ステップ 5（LoRA マージ）はスキップしてステップ 6 へ直接進んでください。

### Q3: LoRA を マージしないで推論したい

非推奨ですが、runtime で可能:

```python
from peft import PeftModel
import torch

base = torch.load("base_checkpoint")
lora = PeftModel.from_pretrained(base, "lora_adapter")
# Use lora directly for inference (no TRT support)
```

**制限:** TRT/ONNX export は LoRA をマージした状態でのみ対応。

### Q4: データセットはどこに置く？

- `input_driving_images/` — 学習用画像
- `input_images/` — テスト用画像

詳細は `finetune/sft/README.md` のデータセット形式を参照。

### Q5: Trainer の config はどこ？

`finetune/sft/configs/` に Hydra 設定テンプレートがあります。必要に応じてカスタマイズしてください。

---

## チェックリスト

- [ ] Python 3.12 venv 作成
- [ ] 依存インストール (`requirements-trt-build.txt` + `peft`)
- [ ] ベースチェックポイント用意
- [ ] Stage1 学習データ準備
- [ ] Stage1 小バッチテスト実行
- [ ] Stage1 本学習実行 → `lora_stage1` 生成
- [ ] Stage1 LoRA マージ → `merged_stage1` 生成
- [ ] Stage2 学習データ準備
- [ ] Stage2 小バッチテスト実行
- [ ] **Stage2 方式選択:**
  - [ ] LoRA パターン: 本学習 → `lora_stage2` 生成 → LoRA マージ → `merged_final` 生成
  - [ ] 全パラメータ SFT パターン: 本学習 → `checkpoint_stage2` 生成（マージ不要）
- [ ] TRT ビルド実行 → `engines_trt/` 生成
- [ ] 推論ノード起動 & 検証

---

## 参考資料

- [ALPAMAYO_FINETUNE_TRT_INTEGRATION.md](./ALPAMAYO_FINETUNE_TRT_INTEGRATION.md) — TRT 詳細, トラブルシュート
- [finetune/sft/README.md](./finetune/sft/README.md) — SFT フレームワーク詳細, API
- [src/alpamayo1_5/models/alpamayo1_5.py](./src/alpamayo1_5/models/alpamayo1_5.py) — モデル実装
- [scripts/build_trt_expert_engine.py](./scripts/build_trt_expert_engine.py) — TRT エクスポートスクリプト

---

**最終更新:** 2026-05-08
