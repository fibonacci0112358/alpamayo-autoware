# Alpamayo 1.5 統合・微調整（SFT LoRA）とTensorRT変換の調査まとめ

最終更新: 2026-05-08

## 概要
- このリポジトリ (`alpamayo-autoware`) は `Alpamayo-1.5` の推論パス（ROSノード、Expert denoiser の ONNX → TensorRT ビルド）を既に含んでいます。
- ローカル別レポジトリ `/home/yutotakeuchi/alpamayo` に SFT（LoRA）と RL の微調整コードが存在します。RL は計算資源の都合で今回は実行不可のため、Stage1（LoRAのみ）のみを実行する想定です。
- 結論: Stage1 LoRA で adapter を学習した後、必ず LoRA をベース重みにマージしてから ONNX/TensorRT エクスポートを実行してください。これにより既存の `scripts/build_trt_expert_engine.py` ワークフローがそのまま使えます。
- runtime は `a1_5_venv`、TensorRT build は `.venv-trt` を使い分けます。`.venv-trt` は build 専用で、Alpamayo1.5 の通常実行には使いません。

## 要点（短く）
- 既存: `src/alpamayo1_5/trt/export.py` と `scripts/build_trt_expert_engine.py` による Expert ONNX export と INT8 quant + TensorRT コンパイルの実装あり。
- SFT: `/home/yutotakeuchi/alpamayo/finetune/sft` に LoRA を使う実装あり。Hydra + HF Trainer ベース。
- 必須: PEFT の LoRA アダプタを `merge_and_unload()` 相当でベースに適用してフルチェックポイントを作成すること（ONNX エクスポート時に PEFT ラッパーが残っていると互換性問題が出るため）。
- TRT ビルド環境: 別途 Python 3.12 仮想環境（README と `scripts/requirements-trt-build.txt` に従う）。ROS 2 Humble のランタイムとは Python バージョンが異なる点に注意。

## 推奨ワークフロー（手順）
1. Stage1 LoRA を通常通り学習する（`finetune/sft/train_hf.py` 等）。
2. 学習後、LoRA アダプタをベース重みにマージして新しい「マージ済チェックポイント」を作成する。
   - 例（PEFT API を使う最小例）:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# base_dir: ベースチェックポイント（元のAlpamayo1.5または変換済チェックポイント）
# lora_dir: LoRA アダプタが保存されたディレクトリ
base = AutoModelForCausalLM.from_pretrained(base_dir, trust_remote_code=True)
peft = PeftModel.from_pretrained(base, lora_dir)

# マージしてメモリからアンロード
merged = peft.merge_and_unload()
merged.save_pretrained("/path/to/merged_checkpoint")
# tokenizer/processor も保存
tokenizer.save_pretrained("/path/to/merged_checkpoint")
```

3. マージ済チェックポイントを `alpamayo-autoware` の推論コードで読み込める形式に配置する（`model_name` をローカルパスにするか、`from_pretrained` でパス指定）。
4. TensorRT エンジンをビルドする専用 venv を用意してアクティブ化する:

```bash
python -m venv .venv-trt
source .venv-trt/bin/activate
pip install -r scripts/requirements-trt-build.txt
```

5. `scripts/build_trt_expert_engine.py` を実行して ONNX エクスポート → quantize → TensorRT コンパイル を実行する（引数に `--expert-onnx-path` や `--model-id` を適切に指定）。

```bash
source .venv-trt/bin/activate
python scripts/build_trt_expert_engine.py \
  --model-id /path/to/merged_checkpoint \
  --output-dir /path/to/trt_output \
  --num-calibration-samples 8
```

6. 生成された TensorRT エンジン（`*.engine`）と ONNX を ROS ノードに供給し、`expert_onnx_path` パラメータを設定してノードを起動する。

## 重要ファイル（参照用）
- `src/alpamayo_ros/alpamayo_ros/alpamayo_node.py`: ROSノード。`Alpamayo1_5.from_pretrained` と `TrtExpertEngine` の組み合わせ。
- `scripts/build_trt_expert_engine.py`: ONNX export / quantize / TRT コンパイルのパイプライン。
- `src/alpamayo1_5/trt/export.py`: ONNX エクスポートユーティリティ。
- `/home/yutotakeuchi/alpamayo/finetune/sft/*`: SFT（LoRA）関連の学習スクリプト。
- `scripts/convert_release_config_to_training.py` (training repo): リリースチェックポイントからトレーニング用チェックポイントへ変換する補助スクリプト。

## 注意点とトラブルシュート
- LoRA をマージせずに ONNX エクスポートを行うと、PEFT ラッパーが残りエクスポートや推論で問題が発生する可能性があります。
- TRT ビルドで使用する `torch` バージョンや `onnxruntime` の互換性に注意。ローカルの `dependency-notes.md` にビルド時のメモを保存しています。
- INT8 quantization でメモリ/OOM が出る場合は `num_calibration_samples` を下げる、または `batch_size` を調整して回避してください。

## 詳細：TRT ビルド手順（例）

1. Python 3.12 の専用 venv を作成・有効化し、必要パッケージをインストールします。ビルドは GPU と CUDA のセットアップが必要です。

```bash
python3.12 -m venv .venv-trt
source .venv-trt/bin/activate
pip install -r scripts/requirements-trt-build.txt
```

2. マージ済チェックポイント（`/path/to/merged_checkpoint`）を `--model-id` に指定してビルドします。`--model-id` は HF ID でもローカルディレクトリパスでも動作します。

```bash
source .venv-trt/bin/activate
python3 scripts/build_trt_expert_engine.py \
  --model-id /path/to/merged_checkpoint \
  --output-dir /path/to/trt_output \
  --num-calibration-samples 8 \
  --calibration-method entropy
```

3. ビルドは次の工程を実行します：推論でデノイザ入力をキャプチャ → ONNX エクスポート → INT8 quantize（SmoothQuant 等）→ TensorRT エンジン生成 → （オプション）ネイティブ vs TRT の簡易検証。

## 実行時の注意とよくあるエラー

- モデルロードで `trust_remote_code=True` を忘れるとカスタム model class を読み込めず失敗します。`scripts/build_trt_expert_engine.py` は `Alpamayo1_5.from_pretrained(..., dtype=torch.bfloat16)` を呼びます。
- ONNX export で `RuntimeError: export failed` のようなエラーが出る場合は、`model.action_in_proj` / `model.expert` / `model.action_out_proj` を `float32` にしてからエクスポートする（既にスクリプト内で行っています）。
- INT8 quantize で calibration 失敗やスコアが悪い場合は `--num-calibration-samples` を 4 に下げる、`--calibration-method` を `percentile` にする、または `--skip-validation` を指定してまずエンジンを作ってから別途検証する。
- CUDA/Trt の互換問題：TensorRT と CUDA の対応表を確認してください（NVIDIA のドキュメント）。また `onnxruntime` の Provider 指定を `CUDAExecutionProvider` にする必要があります。

## トラブルシュート：マージ後に読み込めない場合

- まず `finetune/sft/scripts/verify_merged_checkpoint.py` を使って `Alpamayo1_5.from_pretrained` が成功するか確認してください。

```bash
python3 finetune/sft/scripts/verify_merged_checkpoint.py --merged-dir /path/to/merged_checkpoint
```

- ロードに失敗した場合は tokenizer/config の差分（special token、vocab size、traj token offset）を確認し、必要なら `tokenizer.save_pretrained` と `config.json` をマージ済 checkpoint に入れてください。

## 簡易コマンド集

- マージ（既に LoRA adapter がある場合）:

```bash
python3 finetune/sft/scripts/merge_lora_and_save.py \
  --base-checkpoint /path/to/base_checkpoint \
  --lora-dir /path/to/lora_adapter \
  --output-dir /path/to/merged_output \
  --tokenizer-dir /path/to/tokenizer
```

- 検証（読み込み確認）:

```bash
python3 finetune/sft/scripts/verify_merged_checkpoint.py --merged-dir /path/to/merged_output
```

- TRT ビルド（vENV 内）:

```bash
source .venv-trt/bin/activate
python3 scripts/build_trt_expert_engine.py --model-id /path/to/merged_output --output-dir /path/to/trt_output
```


## 次の推奨アクション
1. （今回）LoRA 学習結果のディレクトリパスとベースチェックポイントのパスを教えてください。マージ用の小スクリプトを作成します。
2. マージ済チェックポイントが用意できたら、私が `scripts/build_trt_expert_engine.py` 実行コマンドを環境向けに整理します（ただしビルドは GPU と専用 venv が必要です）。

---

