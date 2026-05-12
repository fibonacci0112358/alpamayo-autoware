# Stage1 LoRA Fine-Tuning 修正まとめ

このファイルは、Stage1 の LoRA ファインチューニングを動かすまでに行った修正を、時系列と論点ごとにまとめたものです。

## 何を目指していたか

- `finetune/sft/train_hf_1_5.py` から Stage1 の学習を 1 step 以上通す
- Alpamayo のローカル checkpoint を正しく読み込む
- `peft` を使って LoRA を有効化する
- `.pt` 形式の manifest から学習データを読めるようにする
- 途中で出た勾配エラーやモデルロードの不整合を潰す

## 修正していった内容

### 1. モデル設定の土台を整えた

- `alpamayo1_5` の custom config を `AutoConfig` で扱えるようにした
- `Qwen3VL` 系で必要な `rope_scaling` を補正した
- `dtype` / `torch_dtype` の扱いを整理した
- `report_to: []` が `Trainer` で問題にならないように正規化した

### 2. ローカル checkpoint を使うようにした

- ベースモデルをリポジトリ直下の `Alpamayo-1.5-10B` に置き、そこを優先して読むようにした
- checkpoint の実体が単純な upstream `Qwen3VL` ではなく、`alpamayo1_5` 構成であることを確認した
- そのため、直接 `Qwen3VLForConditionalGeneration` として読む経路は fallback に回し、まず Alpamayo の wrapper から読むようにした

### 3. Stage1 用の学習ラッパーを調整した

- `finetune/sft/models/sft_alpamayo_1_5.py` の初期化で、checkpoint から `vlm` を取り出して使うようにした
- `tokenizer` / `special_token_ids` / trajectory tokenizer 系を checkpoint 側に合わせた
- Stage1 では使わない expert / action 系モジュールを freeze した
- LoRA 有効時に、PEFT adapter を trainable に保つよう freeze 条件を調整した

### 4. 勾配が切れる問題を直した

- 以前は backward 時に `element 0 of tensors does not require grad` が出ていた
- 原因は、Stage1 では使わない系統まで trainable / freeze の扱いがずれていたこと
- そこで、Stage1 の実際の学習対象が LoRA 付き VLM に揃うようにした

### 5. データ入力を `.pt` manifest に対応させた

- `PtManifestDataset` を追加し、`manifest.json` から `.pt` サンプルを読むようにした
- `PtManifestCollator` で `alpamayo1_5.helper.create_message()` 相当の形に変換した
- `processor.apply_chat_template()` を通して multimodal 入力を作るようにした
- trajectory 系テンソルも batch で保てるようにした

### 6. Stage1 の起動設定を調整した

- `dataloader_num_workers` を `0` にして worker 起因の負荷を下げた
- local base model があればそれを優先するようにした
- action space / diffusion / projection 系の default config を補った

### 7. 途中で不足していた依存関係を整えた

- `peft` が入っていなかったため、最初は LoRA が無効化されていた
- その後 `peft` を有効化し、LoRA 学習経路に入ることを確認した

## 確認できたこと

- 1 step の学習が回った
- backward エラーは再現しなくなった
- trainable params が LoRA 分だけに絞られた
- checkpoint と tokenizer の保存まで完了した

## 直近の実行結果

- 実行コマンド:

```bash
source /home/yutotakeuchi/alpamayo-autoware/a1_5_venv/bin/activate
python finetune/sft/train_hf_1_5.py +experiment=stage1 data.local_dir=null data.manifest_path=206_dataset_ver1/manifest.json lora.use_lora=true training.max_steps=1 training.dataloader_num_workers=0
```

- 結果:
  - `train_loss` が出て 1 step 完了
  - `Saved checkpoint and tokenizer to outputs/stage1`

## 今後やるなら

- `training.max_steps` を増やして少し長めに回す
- 必要なら `resume_from_checkpoint` を追加して途中再開できるようにする
- 学習率や batch size を本番寄りに詰める
