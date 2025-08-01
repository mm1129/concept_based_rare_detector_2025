## スクリプト概要

`detect_outliers_single_folder` を中心に、NuScenes 画像から YOLOv8 で切り出した物体を CLIP 埋め込み・LOF アウトライア検出・キャプション解析し、特殊クラス・異常サンプルを抽出するパイプライン。

### 主な機能
1. **画像読み込み & シード固定** (`set_seed`)
2. **2D 検出 & 切り抜き** (`DetectorYOLO.detect_and_crop`)
3. **CLIP 埋め込み取得** (`CLIPFeatureExtractor`)
4. **特徴可視化** (`visualize_tsne`)
5. **LOF によるアウトライア検出** (`predict_outliers_lof`, `detect_class_aware_outliers`)
6. **ターゲットクラス抽出** (`get_optimized_target_classes`)
7. **多様性フィルタリング** (`avoid_similar_images`)
8. **Qwen2-VL を用いたキャプション生成** (`generate_descriptions_batch`)
9. **結果の保存 & ディレクトリ構造管理**
10. **ユーティリティ (日本語フォント設定, バッチサイズ最適化)**

---
## モジュール別詳細

### 1. `set_seed(seed: int)`
- **役割**: Python/NumPy/Torch の乱数シードを固定し、再現性を担保する。
- **副作用**: `torch.backends.cudnn.deterministic=True` に設定し、ベンチマークモードをオフ。

### 2. `DetectorYOLO`
- **用途**: Ultralytics `YOLO(model_path)` を用い、信頼度・IoU閾値で検出 → 切り抜き
- **主なメソッド**:
  - `detect_and_crop(image_path, out_dir, conf_thres)`:
    1. 画像ロード → リサイズ（最大1280px）
    2. `self.model(img, conf=conf_thres, iou=0.5, max_det=100, verbose=False)`
    3. 信頼度上位 50 箇所をスコアソート
    4. 元解像度へ座標変換 → PIL でクロップ → `out_dir` に保存
    5. `(出力パス, クラス名, 信頼度, bbox, 元画像パス)` のリストを返却
- **パラメータ**:
  - `model_path`: `yolov8x.pt`（環境変数で切替可）
  - `conf_thres=0.25`, `iou=0.5`, `max_objects=50`

### 3. `CLIPFeatureExtractor`
- **用途**: CLIP ベースの特徴量取得 & L2 正規化
- **初期化**:
  - 環境変数 `EMBEDDER` でモデル切替 (`clip_l14` / `base32`)
- **メソッド**:
  - `get_image_embedding(path)` → `(dim,)` の numpy 配列
  - `get_image_embeddings_batch(paths, batch_size)` → 正規化済みベクトルの配列
  - `get_text_embedding(text_list)` → テキスト埋め込み配列

### 4. `parse_text_with_probability(base_text, candidate_labels, clip_extractor)`
- CLIP テキスト埋め込み同士のコサイン類似度を計算し、0 以上を確率としてソート。ラベル → スコア辞書を返却。

### 5. `generate_descriptions_batch(image_paths, model, processor, batch_size, detected_classes)`
- **用途**: Qwen2-VL を使い、特定クラスの切り抜きに対して詳細キャプションを生成
- **流れ**:
  1. メモリ最適化のバッチサイズ調整
  2. 各パスごとに: YOLO クラス判定 → ターゲット外は簡易キャプション
  3. ターゲット内は base64 でエンコード → OpenAI API 呼び出し
  4. `torch.cuda.amp.autocast` で混合精度生成 → テキスト抽出
- **注意**:
  - 100 枚ごとに GPU キャッシュクリア
  - 失敗時は `None` を挿入

### 6. 可視化: `visualize_tsne(features, labels, title)`
- scikit-learn TSNE → 2D 埋め込み → Matplotlib 散布図出力

### 7. アウトライア検出
- `predict_outliers_lof(features, n_neighbors, contamination)`:
  - `LocalOutlierFactor.fit_predict` → inlier=1, outlier=-1 → `(labels==-1)` を int 化
- `detect_class_aware_outliers(features, class_names, ...)`:
  - クラス毎に LOF を適用 → デフォルト `min_samples=10`
  - 結果を OR して返却

### 8. `get_optimized_target_classes()`
- 基本 [`construction_vehicle`, `bicycle`, `motorcycle`, `trailer`] + 類似クラス一覧を返す

### 9. `avoid_similar_images(image_paths, features, similarity_threshold, max_per_cluster)`
- コサイン類似度行列から閾値以上を同クラスタと見做し、1 クラスタあたり最大数をサンプル。

### 10. メイン: `detect_outliers_single_folder(...)`
- **フロー**:
  1. 出力ディレクトリ & サブディレクトリ構築
  2. シグナルハンドラ, フォント設定, シード固定
  3. 画像ファイル一覧取得 (シャッフル + 最大制限)
  4. DetectorYOLO で全ファイル検出・切り抜き → `all_cropped_info`
  5. CLIP 埋め込み抽出 → 有効ベクトル filter
  6. TSNE 可視化 (実験用)
  7. LOF グローバル & クラス別 → `outlier_flags`
  8. ターゲットクラス抽出 → `target_indices`
  9. 組み合わせ & 多様性フィルタリング → `process_indices`
 10. Qwen2-VL キャプション生成 → `final_results`
 11. JSON 保存, 統計計算, 一時ファイル削除

---


# `detect_improved.py`の実行結果ディレクトリ構造

`detect_improved.py`を実行すると、以下のようなディレクトリ構造とファイルが生成されます：

```
{output_dir}/detection_{timestamp}/
│
├── cropped_objects/                  # 検出されたオブジェクトの切り抜き画像（cleanup_temp_files=Trueの場合は最終的に削除）
│   └── {元画像名}_{インデックス}_{クラス名}.jpg
│
├── outliers/                         # アウトライア（異常）と判定されたオブジェクトの元画像と情報
│   ├── {元画像名}.jpg                # 元の画像ファイル
│   ├── {切り抜き画像名}.jpg          # 切り抜き画像（save_crops=Trueの場合）
│   ├── {切り抜き画像名}_desc.txt     # 説明テキスト（save_descriptions=Trueの場合）
│   └── {切り抜き画像名}_probs.png    # 確率プロット（save_probability_plots=Trueの場合）
│
├── inliers/                          # インライア（正常）と判定されたオブジェクトの元画像と情報
│   ├── {元画像名}.jpg                # 元の画像ファイル
│   ├── {切り抜き画像名}.jpg          # 切り抜き画像（save_crops=Trueの場合）
│   ├── {切り抜き画像名}_desc.txt     # 説明テキスト（save_descriptions=Trueの場合）
│   └── {切り抜き画像名}_probs.png    # 確率プロット（save_probability_plots=Trueの場合）
│
├── final_outliers/                   # 最終的なアウトライア（キャプション解析後）の元画像
│   └── {元画像名}.jpg                # 元の画像ファイル
│
├── targets/                          # 特定のターゲットクラスごとのディレクトリ
│   ├── construction_vehicle/         # 建設車両クラスの画像
│   │   └── {元画像名}.jpg
│   ├── bicycle/                      # 自転車クラスの画像
│   │   └── {元画像名}.jpg
│   ├── motorcycle/                   # バイククラスの画像
│   │   └── {元画像名}.jpg
│   └── trailer/                      # トレーラークラスの画像
│       └── {元画像名}.jpg
│
├── detection_results.json            # 全検出結果のJSON
├── {target_class}_results.json       # 各ターゲットクラスの検出結果JSON
├── statistics.json                   # 統計情報
├── timing.txt                        # 処理時間の記録
├── interim_results.json              # 中間結果（処理途中の保存）
└── process_checkpoint.json           # チェックポイント情報（中断時の再開用）
```

## 主要ファイルの内容

### 1. `{切り抜き画像名}_desc.txt`
説明テキストファイルには以下の情報が含まれます：
```
Sample Token: {サンプルID}
Is Outlier: {True/False}
YOLO Class: {YOLOが検出したクラス}
Target Matches: {ターゲットクラスとの一致}
Final outlier (Top concept not common): {True/False}
Cropped Image Path: {切り抜き画像のパス}
Original Image Path: {元画像のパス}
Confidence={信頼度}, bbox={バウンディングボックス座標}

Generated caption:
{生成されたキャプション}

Top concept: {最も類似度の高いコンセプト} (in common classes: {True/False})
```

### 2. `detection_results.json`
検出結果のJSONファイルには以下の情報が含まれます：
```json
[
  {
    "path": "切り抜き画像のパス",
    "original_path": "元画像のパス",
    "sample_token": "サンプルID",
    "class_name": "YOLOが検出したクラス",
    "confidence": 0.95,
    "bbox": [x1, y1, x2, y2],
    "is_outlier": true,
    "outlier_code": 1,
    "top_concept": "最も類似度の高いコンセプト",
    "is_final_outlier": true,
    "is_yolo_potential": true,
    "target_matches": ["一致したターゲットクラス"],
    "description": "生成されたキャプション"
  },
  // 他の検出結果...
]
```

### 3. `statistics.json`
統計情報のJSONファイルには以下の情報が含まれます：
```json
{
  "total_objects": 1250,
  "processed_objects": 320,
  "outlier_count": 45,
  "final_outlier_count": 28,
  "target_counts": {
    "construction_vehicle": 12,
    "bicycle": 35,
    "motorcycle": 18,
    "trailer": 5
  },
  "class_distribution": {
    "car": 850,
    "pedestrian": 230,
    "traffic_light": 95,
    // 他のクラス...
  }
}
```

## 注意点

1. `--minimal_io`オプションを指定した場合や`--max_images`を指定しない場合は、`save_crops`、`save_descriptions`、`save_probability_plots`が自動的に`False`になり、対応するファイルは生成されません。

2. `--cleanup_temp`オプションを指定した場合、処理完了後に`cropped_objects`ディレクトリは削除されます。

3. 処理中に中断された場合、`process_checkpoint.json`を使用して次回実行時に続きから処理を再開できます。
