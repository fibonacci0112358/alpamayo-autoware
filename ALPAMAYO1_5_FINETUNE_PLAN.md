# Alpamayo 1.5 用 Fine-tuning 実装計画

最終更新: 2026-05-08

## 目的
この文書は、`/home/yutotakeuchi/alpamayo` にある既存の finetune 実装を参考にしつつ、`Alpamayo 1.5` 向けの学習コードをどう作るかを整理したものです。最終目標は、自前データに対して Stage1 の VLM SFT と Stage2 の action/expert SFT を実装し、その後に既存の `alpamayo-autoware` 側の推論・TensorRT 変換パイプラインへ自然に接続できることです。

## まず押さえる前提
- `alpamayo-autoware` 側には `Alpamayo1_5` の推論実装があり、ROS ノードと TRT 用の expert export/runner も既に用意されています。
- したがって、学習コードの主眼は「学習済み重みの作成」であり、推論ランタイムそのものを新規設計する必要はありません。
- 一方で、既存の `finetune` 実装は `alpamayo_r1` 系の名前空間・config・data pipeline を前提にしているため、`Alpamayo1.5` にはそのままでは流用しきれません。

## アーキテクチャ上の差分認識
以下の理解で進めるのが安全です。

- VLM 部分は「同じ思想」でも、実際には config / tokenizer / processor / RoPE / generation 周りの前提が異なる可能性がある。
- Stage1 は主に VLM 側の SFT、Stage2 は action/expert 側の SFT と考えるのが自然です。
- action 生成部分は、`action_space`、`diffusion`、`action_in_proj`、`action_out_proj`、expert 側の position id / attention mask の組み立てが変わるため、Stage2 の移植対象として最重要です。
- つまり、移植の順番は「VLM を完全に差し替える」ではなく、「Alpamayo 1.5 のモデル定義に合わせて Stage1/Stage2 の trainable wrapper と loss を再実装する」が正解です。

## 実装方針
### 1. 既存コードの対応表を作る
まず、`alpamayo_r1` 系の学習コードを `Alpamayo 1.5` 側のクラスへ対応付けます。

- モデル本体
  - 既存: `alpamayo_r1.models.alpamayo_r1.AlpamayoR1`
  - 1.5: `alpamayo1_5.models.alpamayo1_5.Alpamayo1_5`
- 設定
  - 既存: `alpamayo_r1.config.*`
  - 1.5: `alpamayo1_5.config.Alpamayo1_5Config`
- 学習用 wrapper
  - 既存: `TrainableAlpamayoR1`
  - 1.5: 新規に `TrainableAlpamayo1_5` を作る
- データ前処理
  - 既存: `PAIDataset` / `collate_fn` / trajectory token 化
  - 1.5: 同等のデータ処理を保ちつつ、1.5 側の special token / vocab / expert 入力に合わせる

### 2. 学習コードを新規に分ける
`alpamayo_r1` のファイルを無理に上書きせず、`alpamayo1_5` 向けに新規ファイルを作る方が安全です。

推奨ファイル構成の例:

- `finetune/sft/models/sft_alpamayo_1_5.py`
- `finetune/sft/configs/sft_alpamayo_1_5.yaml`
- `finetune/sft/train_hf_alpamayo_1_5.py`
- 必要なら `finetune/sft/data/` 以下に 1.5 専用の preprocess を追加

### 3. まずは Stage1 を通し、次に Stage2 を移植する
最初に Stage1 の VLM SFT を通し、その後に Stage2 の action/expert SFT を移植します。RL は今回は実装対象外です。

Stage1 で必要なこと:
- VLM の forward と loss が動くこと
- trajectory token の埋め込みが正しく行われること
- 小規模データで過学習確認ができること

Stage2 で必要なこと:
- action/expert 側の forward と loss が動くこと
- `action_space` と `diffusion` の入出力が 1.5 の config と一致していること
- expert 側の position id / attention mask / past key values が推論と一致していること
- 小規模データで action 予測が変化することを確認できること

### 4. LoRA を前提にする
今回は軽量に進める前提なので、LoRA を標準手段にします。

学習時の基本方針:
- ベースモデルは freeze し、LoRA のみ学習する
- 学習後は `merge_and_unload()` 相当で重みをマージする
- マージ済 checkpoint と tokenizer / processor をセットで保存する

これにより、後段の `scripts/build_trt_expert_engine.py` に接続しやすくなります。

## 実装の優先順位
### 優先度 1: モデル定義
最初に作るべきなのは Stage1 / Stage2 それぞれの trainable wrapper です。

最低限必要な要素:
- `forward()` で `input_ids`, `ego_history_xyz`, `ego_history_rot`, `ego_future_xyz`, `ego_future_rot` を受ける
- trajectory token を 1.5 の config に従って fuse する
- VLM loss と action loss を分けて計算する
- Stage1 は VLM loss のみ、Stage2 は action/expert loss を主に扱う
- inference で使う `sample_trajectories_from_data()` を 1.5 側の実装に合わせる

### 優先度 2: config
`Alpamayo1_5Config` の学習用 config を作り、Stage1 と Stage2 を分けて明示します。

- `vlm_name_or_path`
- `traj_vocab_size`
- `tokens_per_history_traj`
- `tokens_per_future_traj`
- `action_space_cfg`
- `diffusion_cfg`
- `action_in_proj_cfg`
- `action_out_proj_cfg`
- `expert_cfg`
- `expert_non_causal_attention`
- LoRA の有無と target modules
- Stage1 用の lr / freeze 方針
- Stage2 用の lr / freeze 方針

### 優先度 3: data pipeline
既存データセットをそのまま使えそうに見えても、以下を確認します。

- 画像・テキスト・軌跡のサンプル形式が 1.5 の processor 期待値と一致しているか
- history / future trajectory の shape が 1.5 側の tokenization に合っているか
- `generation_mode` と training mode の分岐が正しいか

### 優先度 4: 保存と変換
学習後の成果物は、推論側が読みやすい形に揃えます。

- checkpoint
- processor / tokenizer
- config.json
- 必要に応じて LoRA adapter 別保存

## 進め方の具体的な順序
1. `Alpamayo 1.5` の Stage1 用 `TrainableAlpamayo1_5` を新規実装する。
2. 小さな固定サンプルで Stage1 の forward を通し、loss が計算できることを確認する。
3. Stage1 の LoRA を有効にして 1 バッチ学習が回ることを確認する。
4. 小規模データで Stage1 の過学習テストを行い、出力が変化することを確認する。
5. Stage1 の重みをベースにして Stage2 用 wrapper を実装する。
6. Stage2 の action/expert loss が計算できることを確認する。
7. Stage2 まで学習した checkpoint を保存する。
8. `alpamayo-autoware` 側でロードできるかを確認する。
9. 既存の TRT export パイプラインに接続する。

## 失敗しやすいポイント
- VLM のベースモデル名や tokenizer が 1.5 と一致していない
- special token の追加順が変わり、vocab size がずれる
- trajectory token の開始位置がずれて、labels の計算が壊れる
- expert 側の attention mask / position ids が inference と学習で不一致
- LoRA をマージせずに export しようとして失敗する

## 実装上の判断基準
次の状態になっていれば、1.5 向け fine-tuning はほぼ正しい方向に進んでいます。

- 学習コードが `alpamayo1_5` 名前空間に揃っている
- config と tokenizer の初期化が 1.5 の推論コードと同じ前提で動く
- Stage1 / Stage2 それぞれで small batch の forward / loss / backward が安定している
- マージ済 checkpoint が `Alpamayo1_5.from_pretrained` で読める
- その checkpoint を元に TRT export が走る

## まず作るべき最小実装
最小構成では以下だけあれば十分です。

- 1.5 用の `TrainableAlpamayo1_5` class
- 1.5 用の SFT config
- 1.5 用の学習エントリポイント
- 小さな動作確認用サンプル
- LoRA merge スクリプト

## 次の作業提案
1. まずは `alpamayo_r1` の Stage1 学習コードをベースに、1.5 用 wrapper の雛形を作る。
2. その後、Stage2 の action/expert 学習コードを別 wrapper として追加する。
3. 最後に LoRA merge と TRT export まで通す。
