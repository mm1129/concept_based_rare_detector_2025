# detect_outliers_single_folder.py - 実装解説

## ファイル概要

`detect_outliers_single_folder.py`は、単一フォルダに格納された画像から希少物体や異常物体を検出するための実装です。1回の実行で一連の処理を完結させ、異常検出の結果を生成します。

## 主要な機能

1. **物体検出と切り抜き**：YOLOv8を使用して画像から物体を検出し、個別の切り抜き画像として保存
2. **特徴抽出**：CLIPモデルを使用して切り抜かれた物体画像から特徴ベクトルを抽出
3. **異常検知**：Isolation Forestを使用して特徴空間における異常を検出
4. **キャプション生成**：Qwen2-VLモデルを使用して物体の詳細な説明を生成
5. **概念マッチング**：生成されたキャプションと概念リストを比較して意味的な異常を検出
6. **結果の出力と可視化**：検出結果の保存、統計情報の生成、t-SNE可視化の作成

## コードの構造と流れ

### 1. インポートと初期設定

```python
# 必要なライブラリのインポート
import os, json, time, torch, argparse
from PIL import Image
import numpy as np
from sklearn.ensemble import IsolationForest
# その他必要なインポート
```

### 2. 画像エンコード関数

```python
def encode_image(image_path):
    """画像をBase64エンコードする関数"""
    # 画像読み込みとBase64エンコード処理
```

### 3. デバイス取得関数

```python
def get_device():
    """利用可能なデバイス（GPU/CPU）を取得する関数"""
    # GPUが利用可能かチェックしてデバイスを返す
```

### 4. 物体検出クラス（DetectorYOLO）

```python
class DetectorYOLO:
    """YOLOv8を使用した物体検出クラス"""
    
    def __init__(self, model_path="yolov8l.pt", device=None):
        """モデルの初期化"""
        # YOLOモデルのロードと初期設定
        
    def detect_and_crop(self, image_path, out_dir="cropped_objects", conf_thres=0.3):
        """画像から物体を検出して切り抜く関数"""
        # 画像の読み込みと前処理
        # 物体検出の実行
        # 検出された物体の切り抜きと保存
        # 検出結果の返却
```

### 5. 特徴抽出クラス（CLIPFeatureExtractor）

```python
class CLIPFeatureExtractor:
    """CLIPを使用した特徴抽出クラス"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """モデルの初期化"""
        # CLIPモデルとプロセッサのロード
        
    def get_image_embedding(self, image_path):
        """単一画像から特徴ベクトルを抽出する関数"""
        # 画像読み込みと前処理
        # 特徴抽出の実行
        # 特徴ベクトルの返却
        
    def get_image_embeddings_batch(self, image_paths, batch_size=32):
        """複数画像からバッチ処理で特徴ベクトルを抽出する関数"""
        # バッチ処理による効率的な特徴抽出
        
    def get_text_embedding(self, text_list):
        """テキストから特徴ベクトルを抽出する関数"""
        # テキスト処理と特徴抽出
```

### 6. テキスト解析関数

```python
def parse_text_with_probability(base_text, candidate_labels, clip_extractor):
    """テキストと概念リストの類似度を計算する関数"""
    # テキストと概念リストの特徴抽出
    # コサイン類似度の計算
    # 類似度の高い概念の抽出と返却
```

### 7. キャプション生成関数

```python
def generate_descriptions_batch(image_paths, model, processor, batch_size=8, detected_classes=None):
    """画像バッチからキャプションを生成する関数"""
    # バッチサイズの最適化
    # 画像の前処理
    # Qwen2-VLモデルによるキャプション生成
    # 進捗表示と結果返却
```

### 8. 可視化関数

```python
def visualize_tsne(features, labels=None, title="t-SNE Visualization"):
    """特徴ベクトルのt-SNE可視化を生成する関数"""
    # t-SNE次元削減の実行
    # 可視化プロットの生成
```

### 9. 異常検知関数

```python
def train_isolation_forest(features, contamination=0.1):
    """Isolation Forestモデルの学習関数"""
    # Isolation Forestの初期化と学習
    
def predict_outliers(features, iso_model):
    """異常値の予測関数"""
    # 学習済みモデルによる予測
```

### 10. シード設定とバッチサイズ最適化

```python
def set_seed(seed=42):
    """乱数シードを設定する関数"""
    # 再現性のためのシード設定
    
def get_optimal_batch_size(initial_size=8, min_size=1):
    """利用可能なGPUメモリに基づいてバッチサイズを最適化する関数"""
    # GPUメモリの取得と最適なバッチサイズの計算
```

### 11. メイン処理関数

```python
def detect_outliers_single_folder(
    images_folder,
    output_dir,
    qwen_model_size="2B",
    contamination=0.1,
    target_classes=None,
    common_classes=None,
    concept_list=None,
    save_crops=True,
    save_descriptions=True,
    save_probability_plots=True,
    cleanup_temp_files=False,
    max_images=None,
    seed=42
):
    """単一フォルダからの異常検出のメイン処理関数"""
    
    # 1. 初期化と設定
    set_seed(seed)
    device = get_device()
    start_time = time.time()
    
    # 出力ディレクトリの作成
    # 設定の保存
    
    # 2. モデルのロード
    # YOLOモデルの初期化
    # CLIPモデルの初期化
    # Qwen2-VLモデルの初期化（必要な場合）
    
    # 3. 画像パスの収集
    # 画像リストの取得と制限（max_imagesが設定されている場合）
    
    # 4. 物体検出と切り抜き
    # 各画像からの物体検出
    # 切り抜き画像の保存
    # 検出結果の収集
    
    # 5. 特徴抽出
    # 切り抜き画像からのCLIP特徴抽出
    
    # 6. t-SNE可視化
    # 特徴の次元削減と可視化
    
    # 7. 異常検知
    # Isolation Forestによる異常検知
    # 異常値と正常値の分類
    
    # 8. クラス情報の収集
    # 検出されたクラスの集計
    # 目標クラスの抽出（target_classesが設定されている場合）
    
    # 9. キャプション生成
    # Qwen2-VLによる物体説明の生成
    
    # 10. 概念マッチング
    # キャプションと概念リストの意味的類似度分析
    
    # 11. 結果の保存
    # 異常オブジェクトの保存
    # 結果JSONの生成
    # 統計情報の生成
    
    # 12. 一時ファイルのクリーンアップ
    # cleanup_temp_filesが有効な場合、不要なファイルを削除
    
    # 13. 処理時間の出力
    # 総処理時間の計算と表示
    
    return detection_results
```

### 12. メイン実行部分

```python
def main():
    """コマンドライン引数を解析してメイン関数を実行"""
    parser = argparse.ArgumentParser(description="単一フォルダからの異常物体検出")
    
    # コマンドライン引数の設定
    # images_folder, output_dir, qwen_model_sizeなどの引数
    
    args = parser.parse_args()
    
    # 引数を使用してdetect_outliers_single_folder関数を呼び出し
    
if __name__ == "__main__":
    main()
```

## 実装の工夫点

1. **メモリ効率の最適化**
   - 動的バッチサイズ調整によるGPUメモリの効率的利用
   - 不要なテンソルの積極的な解放
   - サブバッチ処理による大規模データセット対応

2. **エラー処理の強化**
   - 破損画像や読み込みエラーへの対応
   - 処理失敗時の例外処理と継続実行保証
   - 詳細なエラーログ記録

3. **進捗管理と可視化**
   - 各処理ステップの詳細な進捗表示
   - 残り時間予測による処理計画支援
   - 視覚的に理解しやすい結果可視化

4. **フレキシブルな設定**
   - 多様なコマンドライン引数による柔軟な処理制御
   - クラス指定による特定物体への注目
   - メモリと処理速度のトレードオフ調整オプション

## 使用例

```bash
# 基本的な使用方法
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --output_dir ./results \
  --qwen_model_size 2B

# 特定クラスへの注目
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --output_dir ./results \
  --target_classes "motorcycle bicycle scooter" \
  --common_classes "car truck bus pedestrian"

# メモリ使用量の最適化
python detect_outliers_single_folder.py \
  --images_folder ./your_images \
  --output_dir ./results \
  --qwen_model_size 2B \
  --no_save_crops \
  --no_save_descriptions \
  --max_images 500
```

## 出力結果

処理が完了すると、指定された出力ディレクトリに以下のファイルとフォルダが生成されます：

```
results/
  ├── outlier_detection_[タイムスタンプ]/
  │   ├── config.json            # 実行設定
  │   ├── cropped_objects/       # 切り抜き物体画像
  │   ├── outliers/              # 異常物体
  │   ├── inliers/               # 正常物体
  │   ├── [target_class]/        # 指定クラスの物体
  │   ├── tsne_visualization_class.png  # t-SNE可視化
  │   ├── outlier_detection_result.png  # 異常検出結果
  │   ├── detection_results.json # 詳細結果
  │   ├── statistics.json        # 統計情報
  │   └── timing.txt             # 処理時間情報
```

## まとめ

`detect_outliers_single_folder.py`は単一コマンドで完結する異常物体検出パイプラインを提供します。YOLOv8、CLIP、Qwen2-VLという強力なモデルを組み合わせ、視覚的特徴と意味的理解の両面から異常を検出します。柔軟な設定オプションと詳細な結果出力により、様々な用途に対応可能です。 