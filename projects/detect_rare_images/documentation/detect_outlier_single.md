## スクリプト概要

このスクリプトは、指定フォルダ内の画像から **YOLOv8** による 2D 検出 → 切り抜き → **CLIP** 埋め込み → **IsolationForest** / LOF + 独自フィルタで希少・異常サンプルを選別 → **Qwen2-VL** によるキャプション生成 を一貫して行うパイプラインです。さらに **並列処理** / **サンプリング** / **早期フィルタリング** / キャプションスキップ 等多彩なオプションを備え、大規模データでも柔軟に動作します。

---

### 主な機能と流れ

1. **引数パース & 設定保存**

   - `--sampling_rate`, `--parallel`, `--skip_visualization` などオプションを argparse で受け取る。
   - 設定を JSON (`config.json`) に保存。

2. **初期処理**

   - `setup_japanese_font()` で matplotlib 日本語フォント設定
   - `set_seed()` で再現性確保
   - 出力ディレクトリおよびサブディレクトリ (`cropped_objects`, `final_outliers` など) の作成

3. **画像リスト取得 & サンプリング**

   - 対象拡張子の画像を列挙→シャッフル
   - `sampling_rate < 1.0` ならランダムサンプリングで枚数削減（大規模時の高速化）

4. **並列オブジェクト検出 & 切り抜き**

   - `DetectorYOLO` 初期化
   - バッチ(50枚)単位に分割し、`ProcessPoolExecutor` で `process_image_batch` を並列実行
   - 結果：`all_cropped_info[(crop_path, class_name, conf, bbox, orig_path), ...]`

5. **早期フィルタリング**

   - 検出済みオブジェクトから `is_potential_rare_class` で希少クラスを優先抽出
   - 条件に満たない場合、ランダムで一定数追加し多様性を確保
   - 絞り込み後の `all_cropped_info` を次段階へ

6. **CLIP 特徴抽出**

   - `CLIPFeatureExtractor` を `clip-vit-base-patch32` で初期化
   - `get_image_embeddings_batch` でバッチ(32枚)単位で GPU 上で抽出
   - 成功したインデックスのみ `valid_features` / `valid_cropped_info` に保存

7. **t-SNE 可視化**（スキップ可）

   - `skip_visualization=False` の場合のみ実行
   - `TSNE(n_components=2)` で 2D に射影し、Matplotlib 散布図を保存

8. **IsolationForest アウトライア検出**

   - `train_isolation_forest` でモデル学習 → `predict_outliers` でラベル取得 (−1: 異常)
   - `outlier_flags` に 1/0 のマスクを生成

9. **希少クラス再フィルタリング**

   - `filter_rare_classes` で頻度 & キーワードに基づくスコア上位 30% or 最低50件を選択
   - `outlier_flags` と union し、最終 `process_indices` を確定

10. **キャプション生成**（スキップ可）

    - `skip_caption=False` の場合のみ実行
    - `generate_descriptions_batch`:
      - 非希少クラスは簡易キャプション (クラス名)
      - 希少クラスのみ base64 → OpenAI Qwen2-VL コール
      - 並列キャッシュクリア & 例外時フォールバック
    - スキップ時は `is_potential_rare_class` のみでフラグ付け

11. **結果保存**

    - JSON (`detection_results.json`), 各種ディレクトリへのコピー（cropped, inliers, final\_outliers, targets）
    - 統計 (`statistics.json`)、タイミング (`timing.txt`)

12. **後処理**

    - GPU メモリクリア, オブジェクト削除
    - 完了メッセージ出力

---

## 主要コンポーネント

### DetectorYOLO

- **モデル**: Ultralytics YOLOv8 (`yolov8x.pt` 等)
- **detect\_and\_crop**:
  - リサイズ→検出 (`conf=0.25`, `iou=0.45`, `max_det=100`)
  - 信頼度ソート & 上位50切り抜き
  - 元解像度座標に逆変換
  - 結果タプルのリストを返却

### CLIPFeatureExtractor

- **モデル**: `openai/clip-vit-base-patch32`
- **get\_image\_embeddings\_batch**:
  - バッチ & padding 処理
  - GPU 上で特徴量取得
  - メモリ最適化後キャッシュクリア

### IsolationForest

- **train\_isolation\_forest** / **predict\_outliers** でグローバル異常検出

### filter\_rare\_classes

- クラス名頻度 & キーワードによるスコア付け
- 上位 30% or 最低 50 インデックスを希少候補とする

### generate\_descriptions\_batch

- 非希少はクラス名リターン
- 希少のみ詳細に OpenAI GPT-VL 呼び出し
- 10 枚ごとにキャッシュクリア

### 並列化

- **process\_image\_batch**: 画像バッチごとの detect\_and\_crop を副プロセスで実行
- CPU コア数の80%をワーカー数に使用

---

このドキュメントが、主要処理の流れと各ステップの役割を理解する助けになれば幸いです。問題や追加説明が必要な箇所があればお知らせください。

## 実行例: 入力セットの仮定

例えば、`images_folder`に nuScenes の前方カメラ画像1000枚を格納し、以下のオプションで実行した場合を考える:

- `--sampling_rate 0.2` (20%サンプリング)
- `--parallel`
- `--skip_visualization`
- `--skip_caption`
- `--yolo_model yolov8x.pt`
- `--clip_model laion/CLIP-ViT-B-32-laion2B-s34B-b79K`

1. **サンプリング**: シャッフル後 1000 → 200枚をランダム選出
2. **並列検出**: CPUコア数の80%（例:8コア）で50枚×4バッチに分割し YOLOv8x による検出。各画像あたり平均10検出 → 計約2000オブジェクト切り抜き。
3.

   **早期フィルタ**: `is_potential_rare_class` に該当する希少クラス(例:建設車両、自転車)が約500件 → フィルタ後500。多様性確保用に残りのランダム追加100 → 計600オブジェクト。
4. **CLIP埋め込み**: 600件を32バッチ(約19バッチ)でGPU並列処理。全成功 → 600×512次元ベクトル取得。
5. **異常検出**: IsolationForest(contamination=0.1)で学習・予測 → 異常フラグ60件。
6. **希少フィルタ**: `filter_rare_classes` でスコア上位30%(180件)を選出 → 最終 `process_indices`: 異常60 ∪ 希少180 = 240件。
7. **キャプションスキップ**: `skip_caption=True` のため、`is_potential_rare_class` のみで簡易ラベリング。
8. **出力**: 240件のオブジェクト画像を `final_outliers` にコピーし、`config.json`、`statistics.json`、`timing.txt` を出力。

この実行例により、各段階での処理件数や出力量のイメージをつかむことができる。

