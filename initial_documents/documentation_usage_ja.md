# 概念ベース希少物体検出システム - 使用方法ガイド

本ドキュメントでは、概念ベース希少物体検出システムの使用方法、ワークフロー、結果の解釈方法について説明します。

## 1. システムのセットアップ

### 1.1 必要環境

- **Python**: 3.8以上
- **GPU**: CUDA対応GPUを推奨（8GB以上のGPUメモリ）
- **メモリ**: 16GB以上の RAM

### 1.2 必要パッケージのインストール

```bash
# 基本パッケージ
pip install torch torchvision numpy pillow matplotlib scikit-learn

# 物体検出
pip install ultralytics

# 特徴抽出とキャプション生成
pip install transformers accelerate python-dotenv

# 画像処理と可視化
pip install opencv-python seaborn
```



## 2. システムの基本的な使用方法

システムは主に2つの動作モードがあります：

1. **単一フォルダ処理モード**: 1つのフォルダ内の画像を処理して異常を検出
2. **ベースライン学習モード**: 正常サンプルからモデルを学習し、新規データで異常検出

### 2.1 単一フォルダ処理モード

このモードは、特定のフォルダ内の画像を処理して異常オブジェクトを検出します。

```bash
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --output_dir ./results \
  --qwen_model_size 2B \
  --contamination 0.1
```

**主要な引数:**

- `--images_folder`: 処理する画像が格納されているフォルダ
- `--output_dir`: 結果を保存するフォルダ
- `--qwen_model_size`: 使用するQwenモデルのサイズ（2B または 7B）
- `--contamination`: IsolationForestの汚染率（想定される異常値の割合）
- `--target_classes`: 特に注目するクラス（例: "construction_vehicle bicycle"）
- `--common_classes`: 一般的なクラス（異常と見なさない）
- `--max_images`: 処理する最大画像数（大量データでの部分処理に有用）

### 2.2 ベースライン学習モード

このモードでは、「正常」とみなされるサンプルからベースラインモデルを学習し、新規画像に対して異常検出を行います。

```bash
python whole-architecture_batch.py \
  --baseline_folder ./normal_images \
  --test_folder ./new_images \
  --output_dir ./results \
  --qwen_model_size 2B
```

**主要な引数:**

- `--baseline_folder`: 正常サンプルが格納されているフォルダ
- `--test_folder`: 検査する新規画像が格納されているフォルダ
- `--output_dir`: 結果を保存するフォルダ
- `--qwen_model_size`: 使用するQwenモデルのサイズ（2B または 7B）
- `--train_ratio`: トレーニングに使用する画像の割合（0-1）
- `--contamination`: 異常検出の感度パラメータ（0-1）

## 3. 処理ワークフロー

システムは以下のステップで画像を処理します：

### 3.1 単一フォルダ処理モードのワークフロー

1. **初期設定と準備**:
   - 出力ディレクトリの作成
   - 設定情報の保存（config.json）
   - YOLOモデルの初期化

2. **物体検出と切り抜き**:
   - YOLOv8による各画像中の物体検出
   - 検出された物体の切り抜きと保存
   - 信頼度とバウンディングボックス情報の記録

3. **特徴抽出**:
   - CLIPによる切り抜き画像からの特徴ベクトル抽出
   - バッチ処理による効率的な特徴抽出

4. **異常検知**:
   - t-SNEによる特徴の可視化
   - IsolationForestによる異常オブジェクトの検出
   - クラス分布の分析と可視化

5. **高度な意味解析**:
   - Qwen2-VLによる画像キャプション生成
   - キャプションの概念分析（CLIPテキスト埋め込みを使用）
   - 一般的でない概念に基づく追加的な異常判定

6. **結果の整理と出力**:
   - 異常オブジェクトと通常オブジェクトの分類
   - 特定クラスのオブジェクトの抽出
   - 結果の可視化と統計情報の生成

### 3.2 ベースライン学習モードのワークフロー

1. **ベースラインデータ処理**:
   - 正常サンプルからのオブジェクト検出と特徴抽出
   - IsolationForestモデルの学習
   - クラス分布の分析と視覚化

2. **概念リストの生成/拡張**:
   - 検出クラスからの候補ラベル生成
   - 必要に応じたVLMによる概念拡張

3. **新規画像の処理**:
   - 新規画像からのオブジェクト検出と特徴抽出
   - 学習済みモデルによる異常スコア計算
   - キャプション生成と概念マッチング

4. **マルチモーダル統合分析**:
   - 特徴ベースの異常スコアと概念ベースの異常スコアの統合
   - 最終的な異常判定

5. **結果の出力と可視化**:
   - 異常オブジェクトの抽出と保存
   - t-SNEによる特徴空間の可視化
   - 詳細な分析結果のJSON形式での保存

## 4. 出力結果の説明と解釈

システムは以下のような出力を生成します：

### 4.1 ディレクトリ構造

```
results/
  ├── outlier_detection_[タイムスタンプ]/
  │   ├── config.json            # 実行設定のログ
  │   ├── cropped_objects/       # 切り抜かれた物体画像
  │   ├── outliers/              # 異常と判定された物体
  │   ├── inliers/               # 正常と判定された物体
  │   ├── [target_class]/        # 特定クラス（例: bicycle）の物体
  │   ├── tsne_visualization_class.png  # クラス別t-SNE可視化
  │   ├── outlier_detection_result.png  # 異常検出結果の可視化
  │   ├── detection_results.json # 詳細な検出結果
  │   ├── statistics.json        # 統計情報
  │   └── timing.txt             # 処理時間情報
```

### 4.2 主要な出力ファイルの解釈

1. **config.json**:
   - 実行時の設定パラメータを記録
   - 再現性確保のための重要情報

2. **tsne_visualization_class.png**:
   - 特徴空間の2D可視化
   - 色分けされたクラス分布の視覚化
   - クラスター形成やクラス間関係の把握に有用

3. **outlier_detection_result.png**:
   - IsolationForestによる異常検出結果
   - 青点: 正常サンプル、赤点: 異常サンプル
   - 分布の外れ値や孤立点の視覚的確認

4. **detection_results.json**:
   詳細な検出結果が含まれる JSON ファイルで、各オブジェクトに関する情報:
   ```json
   [
     {
       "path": "cropped_objects/image1_0_car.jpg",
       "original_path": "your_images/image1.jpg",
       "class_name": "car",
       "confidence": 0.92,
       "bbox": [100, 150, 300, 400],
       "is_outlier": false,
       "top_concept": "car",
       "is_final_outlier": false,
       "target_matches": [],
       "description": "A red sedan car driving on a city street..."
     },
     {
       "path": "cropped_objects/image2_1_bicycle.jpg",
       "original_path": "your_images/image2.jpg",
       "class_name": "bicycle",
       "confidence": 0.87,
       "bbox": [200, 250, 300, 400],
       "is_outlier": true,
       "top_concept": "unusual_bicycle",
       "is_final_outlier": true,
       "target_matches": ["bicycle"],
       "description": "An unusually modified bicycle with extra attachments..."
     }
   ]
   ```

5. **statistics.json**:
   処理の統計情報を提供:
   ```json
   {
     "total_objects": 534,
     "processed_objects": 87,
     "outlier_count": 53,
     "final_outlier_count": 42,
     "target_counts": {
       "bicycle": 15,
       "construction_vehicle": 8
     },
     "class_distribution": {
       "car": 245,
       "person": 98,
       "bicycle": 15,
       ...
     }
   }
   ```

### 4.3 異常判定の解釈

システムは2つの異常判定基準を使用します:

1. **特徴空間での異常 (`is_outlier`)**:
   - IsolationForestによる判定
   - 特徴空間において他のサンプルから離れた位置にあるかどうか
   - 視覚的な特徴（色、形状、テクスチャなど）に基づく異常性

2. **概念ベースの異常 (`is_final_outlier`)**:
   - キャプションの意味解析に基づく判定
   - トップコンセプトが一般的なクラス（common_classes）に含まれないか
   - 意味的な異常性（珍しい状況や物体の組み合わせなど）

真の異常は、この2つの判定基準のいずれか、または両方を満たすサンプルと考えられます。

## 5. 高度な使用方法とカスタマイズ

### 5.1 概念リストのカスタマイズ

特定ドメインに応じて候補ラベルリストを調整できます:

```python
custom_concept_list = [
    "car", "truck", "bicycle", "pedestrian",
    # ドメイン固有の概念を追加
    "road_construction", "traffic_accident", "street_vendor",
    "road_debris", "fallen_tree", "flooded_area"
]

# コマンドラインから指定
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --concept_list "car truck bicycle pedestrian road_construction traffic_accident"
```

### 5.2 メモリ使用量の最適化

限られたGPUリソースでも実行できるようにメモリ使用量を最適化:

```bash
# メモリ使用量を削減するオプション
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --qwen_model_size 2B \  # 小さいモデルを使用
  --no_save_crops \        # 切り抜き画像を保存しない
  --no_save_descriptions \ # 説明テキストを保存しない 
  --max_images 500         # 処理画像数を制限
```

または、一度に全てを処理せずに分割して処理:

```bash
# 1ステップ目：最初の500画像を処理
python detect_outliers_single_folder.py --images_folder ./your_images --max_images 500 --output_dir ./results_part1

# 2ステップ目：次の500画像を処理
# (ファイル名ソートが異なるため、適切にインデックスを調整する必要があります)
```

### 5.3 特定クラスに焦点を当てた分析

特定の物体クラスに注目して分析:

```bash
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --target_classes "motorcycle bicycle scooter" \
  --common_classes "car truck bus pedestrian"
```

この設定により、二輪車（motorcycle, bicycle, scooter）に関連する異常に特に注目します。

## 6. トラブルシューティング

### 6.1 メモリエラーへの対処

```
CUDA out of memory
```

対処法:
- `--qwen_model_size 2B`を使用（7Bよりメモリ効率が良い）
- `--max_images`でバッチサイズを小さくする
- `--no_save_crops --no_save_descriptions`でI/O操作を減らす
- `--minimal_io`オプションを使用してすべてのI/O削減オプションを有効にする

### 6.2 処理速度の改善

処理が遅い場合:
- GPUの使用を確認（`nvidia-smi`コマンド）
- 画像数を減らす（`--max_images`オプション）
- 軽量モデルの使用（`--qwen_model_size 2B`）
- 進捗表示に注目し、ボトルネックを特定

### 6.3 物体検出精度の問題

物体検出の結果が不十分:
- 検出閾値の調整: `--conf_thres 0.2`（デフォルトは0.3）
- 異なるYOLOモデルの使用（コード内の`model_path`パラメータを変更）
- より高解像度の入力画像の使用

## 7. バッチ処理と自動化

複数のフォルダを連続して処理するスクリプトの例:

```bash
#!/bin/bash

# 複数のデータセットを処理するスクリプト
DATASETS=("dataset1" "dataset2" "dataset3")
OUTPUT_DIR="./all_results"

for dataset in "${DATASETS[@]}"; do
  echo "Processing dataset: $dataset"
  python detect_outliers_single_folder.py \
    --images_folder "./data/$dataset" \
    --output_dir "$OUTPUT_DIR/$dataset" \
    --qwen_model_size 2B
done

echo "All datasets processed. Results saved to $OUTPUT_DIR"
```

## 8. 結果の分析とレポート生成

結果を分析するためのPythonスクリプト例:

```python
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

# 結果ディレクトリ
result_dir = "results/outlier_detection_20230615_12-34-56_JST"

# 検出結果の読み込み
with open(os.path.join(result_dir, "detection_results.json"), "r") as f:
    results = json.load(f)

# 統計情報の読み込み
with open(os.path.join(result_dir, "statistics.json"), "r") as f:
    stats = json.load(f)

# データフレームへの変換
df = pd.DataFrame(results)

# 異常物体の分析
outliers = df[df["is_final_outlier"] == True]
print(f"Total outliers: {len(outliers)}")

# トップコンセプトの分布
concept_counts = outliers["top_concept"].value_counts()
print("Top concept distribution among outliers:")
print(concept_counts.head(10))

# クラスごとの異常割合
class_outlier_ratio = df.groupby("class_name").agg({
    "is_final_outlier": "mean",
    "path": "count"
}).sort_values("is_final_outlier", ascending=False)
print("\nOutlier ratio by class:")
print(class_outlier_ratio.head(10))

# 可視化
plt.figure(figsize=(10, 6))
concept_counts.head(10).plot(kind="bar")
plt.title("Most Common Concepts in Outlier Objects")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "outlier_concepts.png"))
```

このスクリプトは、検出結果から異常物体の概念分布やクラスごとの異常率を分析し、視覚化します。 