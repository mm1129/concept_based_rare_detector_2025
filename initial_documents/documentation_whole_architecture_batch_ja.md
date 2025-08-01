# whole-architecture_batch.py - 実装解説

## ファイル概要

`whole-architecture_batch.py`は、ベースラインデータからモデルを学習し、新しい画像データから異常物体を検出するための2段階アプローチを実装したファイルです。正常サンプルからの学習と、新規データでの検出を一連のパイプラインとして提供します。

## 主要な機能

1. **ベースラインモデル学習**：正常と見なされるデータから特徴空間モデルを構築
2. **新規画像の異常検出**：学習したモデルを使用して新しい画像から異常を検出
3. **概念拡張と生成**：検出されたクラスを基に概念リストを生成・拡張
4. **バッチ処理とメモリ管理**：大規模データセットを効率的に処理するためのメモリ最適化
5. **結果統合と出力**：複数の分析結果を統合し、詳細なレポートを生成

## コードの構造と流れ

### 1. インポートと初期設定

```python
# 必要なライブラリのインポート
import os, sys, json, time, torch, argparse, gc
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
# その他必要なインポート
```

### 2. 画像エンコードとデバイス管理

```python
def encode_image(image_path):
    """画像をBase64エンコードする関数"""
    # 画像読み込みとBase64エンコード処理

def get_device():
    """利用可能なデバイス（GPU/CPU）を取得する関数"""
    # GPUが利用可能かチェックしてデバイスを返す
```

### 3. 物体検出クラス（DetectorYOLO）

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

### 4. 特徴抽出クラス（CLIPFeatureExtractor）

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

### 5. テキスト解析と概念生成

```python
def parse_text_with_probability(base_text, candidate_labels, clip_extractor):
    """テキストと概念リストの類似度を計算する関数"""
    # テキストと概念リストの特徴抽出
    # コサイン類似度の計算
    # 類似度の高い概念の抽出と返却

def generate_description(image_path, model, processor):
    """単一画像の説明を生成する関数"""
    # 画像の読み込みとエンコード
    # 説明生成のためのプロンプト作成
    # モデルによる説明生成
    # 結果の返却

def generate_descriptions_batch(image_paths, model, processor, batch_size=8, detected_classes=None):
    """画像バッチからキャプションを生成する関数"""
    # バッチサイズの最適化
    # 画像の前処理
    # Qwen2-VLモデルによるキャプション生成
    # 進捗表示と結果返却
```

### 6. 可視化と異常検知

```python
def visualize_tsne(features, labels=None, title="t-SNE Visualization"):
    """特徴ベクトルのt-SNE可視化を生成する関数"""
    # t-SNE次元削減の実行
    # 可視化プロットの生成
    
def train_isolation_forest(features, contamination=0.1):
    """Isolation Forestモデルの学習関数"""
    # Isolation Forestの初期化と学習
    
def predict_outliers(features, iso_model):
    """異常値の予測関数"""
    # 学習済みモデルによる予測
```

### 7. ベースラインモデル学習

```python
def train_baseline_model(baseline_folder, qwen_model_size, use_concept_list, concept_list, main_out_dir, lim1=0.8, train_indices=None):
    """ベースラインデータからモデルを学習する関数"""
    
    # 1. 初期化と設定
    # デバイスの取得
    # 出力ディレクトリの作成
    
    # 2. モデルのロード
    # YOLOモデルの初期化
    # CLIPモデルの初期化
    
    # 3. ベースライン画像の処理
    # 画像リストの取得
    # 検出と切り抜き
    # 特徴抽出
    
    # 4. 特徴空間の分析
    # t-SNE可視化
    # Isolation Forestモデルの学習
    
    # 5. クラス情報の収集
    # 検出されたクラスの集計
    # 統計情報の生成
    
    # 6. 概念リストの生成（使用する場合）
    # クラス名から候補概念の生成
    # VLMによる概念拡張
    
    # 7. ベースラインデータの保存
    # 特徴ベクトル、モデル、設定などの保存
    
    return baseline_results
```

### 8. 新規画像検出

```python
def detect_new_images(
    new_images_folder,
    clip_extractor,
    candidate_labels,
    qwen_model_size,
    main_out_dir,
    common_classes,
    baseline_features=None,
    baseline_tsne=None,
    baseline_class_names=None,
    iso_model=None,
    threshold_percentile=95,
    lim1=0.8,
    lim2=1.0,
    save_outliers=True,
    save_outlier_list=True,
    test_indices=None,
    only_additional_candidates=False,
    potential_classes=None,
    save_crops=True,
    save_descriptions=True,
    save_probability_plots=True,
    cleanup_temp_files=False
):
    """ベースラインモデルを使用して新規画像から異常を検出する関数"""
    
    # 1. 初期化と設定
    # 出力ディレクトリの作成
    # 設定の保存
    
    # 2. モデルのロード
    # YOLOモデルの初期化
    # Qwen2-VLモデルの初期化（必要な場合）
    
    # 3. 画像パスの収集
    # 画像リストの取得
    
    # 4. 物体検出と切り抜き
    # 各画像からの物体検出
    # 切り抜き画像の保存
    # 検出結果の収集
    
    # 5. 特徴抽出
    # 切り抜き画像からのCLIP特徴抽出
    
    # 6. 異常検知（特徴ベース）
    # ベースライン特徴との比較
    # Isolation Forestスコアの計算
    # 閾値に基づく異常判定
    
    # 7. t-SNE可視化（ベースラインと新規データの統合）
    # 特徴の次元削減と可視化
    
    # 8. キャプション生成
    # Qwen2-VLによる物体説明の生成
    
    # 9. 概念マッチング（意味ベースの異常検知）
    # キャプションと概念リストの意味的類似度分析
    # 一般的でない概念の検出
    
    # 10. 結果の統合と保存
    # 特徴ベースと意味ベースの異常判定の統合
    # 異常オブジェクトの保存
    # 結果JSONの生成
    
    # 11. 一時ファイルのクリーンアップ
    # cleanup_temp_filesが有効な場合、不要なファイルを削除
    
    return detection_results
```

### 9. メモリ監視と最適化

```python
def monitor_memory():
    """GPUメモリ使用量を監視する関数"""
    # GPUメモリの使用状況を取得して表示

def signal_handler(signum, frame):
    """中断シグナルハンドラ"""
    # プログラム中断時のクリーンアップ処理

def get_optimal_batch_size(initial_size=8, min_size=1):
    """利用可能なGPUメモリに基づいてバッチサイズを最適化する関数"""
    # GPUメモリの取得と最適なバッチサイズの計算
```

### 10. 概念生成と拡張

```python
def generate_feature_dict(class_names, qwen_model_size="7B", prompt_type="important"):
    """クラス名から追加の概念候補を生成する関数"""
    # VLMモデルの初期化
    # クラス名ごとに関連概念の生成
    # 結果の整形と返却
```

### 11. その他のユーティリティ関数

```python
def set_seed(seed=42):
    """乱数シードを設定する関数"""
    # 再現性のためのシード設定
```

### 12. メイン実行部分

```python
def main():
    """コマンドライン引数を解析してメイン処理を実行"""
    parser = argparse.ArgumentParser(description="ベースライン学習と新規画像からの異常検出")
    
    # コマンドライン引数の設定
    # baseline_folder, test_folderなどの引数
    
    args = parser.parse_args()
    
    # 1. ベースラインモデルの学習
    baseline_results = train_baseline_model(
        args.baseline_folder,
        args.qwen_model_size,
        args.use_concept_list,
        args.concept_list,
        args.output_dir,
        args.lim1,
        None  # train_indices
    )
    
    # 2. 新規画像の異常検出
    detect_new_images(
        args.test_folder,
        baseline_results["clip_extractor"],
        baseline_results["candidate_labels"],
        args.qwen_model_size,
        args.output_dir,
        baseline_results["common_classes"],
        baseline_results["features"],
        baseline_results["tsne"],
        baseline_results["class_names"],
        baseline_results["iso_model"],
        args.threshold_percentile,
        args.lim1,
        args.lim2,
        # その他のパラメータ
    )

if __name__ == "__main__":
    main()
```

## 実装の工夫点

1. **2段階アプローチ**
   - ベースライン学習と新規画像検出の明確な分離
   - 各段階での中間結果の保存による再利用性向上

2. **メモリ管理の高度な最適化**
   - サブバッチ処理による大規模データセット対応
   - 定期的なGPUメモリモニタリングとクリーニング
   - シグナルハンドラによる安全な中断処理

3. **マルチモーダル分析の統合**
   - 特徴ベースの異常検知（Isolation Forest）
   - 意味ベースの異常検知（テキスト生成と概念マッチング）
   - 両アプローチの結果統合による精度向上

4. **概念拡張メカニズム**
   - 検出されたクラスからの基本概念生成
   - VLMを活用した関連概念の自動拡張
   - 検出範囲拡大のための概念辞書構築

## 処理フロー

`whole-architecture_batch.py`の処理は、大きく以下の2つのフェーズに分かれています：

### フェーズ1: ベースラインモデルの学習

1. **データ準備**
   - ベースラインフォルダから画像の読み込み
   - YOLOv8による物体検出と切り抜き

2. **特徴抽出と分析**
   - CLIPによる特徴ベクトル抽出
   - クラス分布分析と可視化
   - Isolation Forestモデルの学習

3. **概念リスト生成**
   - 検出クラスからの基本概念生成
   - VLMによる概念拡張（オプション）

4. **結果保存**
   - 学習済みモデル、特徴ベクトル、設定の保存
   - 中間結果の保存（t-SNE可視化など）

### フェーズ2: 新規画像の異常検出

1. **データ処理**
   - 新規画像フォルダからの画像読み込み
   - YOLOv8による物体検出と切り抜き

2. **特徴ベースの異常検知**
   - CLIP特徴ベクトルの抽出
   - ベースライン特徴との比較
   - 学習済みIsolation Forestモデルでの異常スコア計算

3. **意味ベースの異常検知**
   - Qwen2-VLによるキャプション生成
   - キャプションと概念リストのマッチング
   - 一般的でない概念の特定

4. **統合分析と結果出力**
   - 特徴ベースと意味ベースの異常判定の統合
   - 異常オブジェクトの抽出と保存
   - 詳細な検出結果と統計情報の生成

## 使用例

```bash
# 基本的な使用方法
python whole-architecture_batch.py \
  --baseline_folder ./normal_images \
  --test_folder ./new_images \
  --output_dir ./results \
  --qwen_model_size 2B

# 概念リスト生成を有効化
python whole-architecture_batch.py \
  --baseline_folder ./normal_images \
  --test_folder ./new_images \
  --output_dir ./results \
  --use_concept_list \
  --qwen_model_size 7B

# 詳細設定の例
python whole-architecture_batch.py \
  --baseline_folder ./normal_images \
  --test_folder ./new_images \
  --output_dir ./results \
  --qwen_model_size 2B \
  --contamination 0.05 \
  --threshold_percentile 97 \
  --cleanup_temp_files
```

## 出力結果

処理が完了すると、指定された出力ディレクトリに以下のファイルとフォルダが生成されます：

```
results/
  ├── baseline_[タイムスタンプ]/
  │   ├── config.json            # ベースライン設定
  │   ├── cropped_objects/       # ベースラインの切り抜き物体
  │   ├── features.npy           # 特徴ベクトル
  │   ├── iso_model.pkl          # 学習済みIsolation Forest
  │   ├── tsne_visualization.png # t-SNE可視化
  │   └── class_statistics.json  # クラス分布統計
  │
  ├── detection_[タイムスタンプ]/
  │   ├── config.json            # 検出設定
  │   ├── cropped_objects/       # 新規画像の切り抜き物体
  │   ├── outliers/              # 異常判定された物体
  │   ├── combined_tsne.png      # ベースラインと新規データの統合可視化
  │   ├── detection_results.json # 詳細な検出結果
  │   ├── statistics.json        # 検出統計情報
  │   └── timing.txt             # 処理時間情報
```

## まとめ

`whole-architecture_batch.py`は、正常サンプルからベースラインモデルを学習し、そのモデルを使用して新規画像から異常を検出する2段階のパイプラインを提供します。特徴ベースと意味ベースの異常検知を組み合わせることで、視覚的な特性と意味的な理解の両面から異常物体を特定します。メモリ効率の最適化と柔軟な設定オプションにより、様々な規模のデータセットと用途に対応できます。 