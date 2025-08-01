# 概念ベース希少物体検出システム - モデル詳細ドキュメント

## 1. 使用モデルの概要と選定理由

本システムでは3種類の深層学習モデルを組み合わせて使用しています。各モデルの役割と選定理由について説明します。

### 1.1 物体検出モデル：YOLOv8

**選定理由**：
- 高速な推論速度と高い検出精度のバランス
- 多種多様な物体カテゴリに対する汎用性
- リアルタイム処理を視野に入れた軽量設計
- 継続的な改良とコミュニティサポート

**使用バージョン**：
- YOLOv8l（large）：速度と精度のバランスが取れたモデル
- COCO データセットで事前学習済み（80クラス対応）

### 1.2 特徴抽出モデル：CLIP

**選定理由**：
- 画像とテキストの共通埋め込み空間の学習
- 40億のテキスト-画像ペアによる大規模事前学習
- ゼロショット物体認識能力
- 特徴の汎用性と転移学習の容易さ

**使用バージョン**：
- openai/clip-vit-base-patch32：512次元の特徴ベクトルを生成
- Vision Transformer (ViT) アーキテクチャを採用

### 1.3 視覚言語モデル：Qwen2-VL

**選定理由**：
- 高度な画像理解と詳細なテキスト生成能力
- マルチモーダル情報の統合的理解
- 様々なサイズの画像入力に対応
- 多様な視覚タスクに対する高い性能

**使用バージョン**：
- Qwen2-VL-2B（標準）：計算リソースが限られた環境向け
- Qwen2-VL-7B（大規模）：より高度な理解と生成能力が必要な場合

## 2. 各モデルの役割と実装詳細

### 2.1 YOLOv8：物体検出と抽出

```python
class DetectorYOLO:
    def __init__(self, model_path="yolov8l.pt", device=None):
        # モデルの初期化
        self.model = YOLO(model_path)
        self.model.to(self.device)
```

**実装詳細**：

1. **動的リサイズと処理効率化**
   ```python
   max_size = 1280
   if width > max_size or height > max_size:
       if width > height:
           new_width = max_size
           new_height = int(height * (max_size / width))
       else:
           new_height = max_size
           new_width = int(width * (max_size / height))
       img = img.resize((new_width, new_height), Image.LANCZOS)
   ```
   - 1280pxを超える画像は処理効率向上のために動的にリサイズ
   - アスペクト比を維持しながらスケーリング
   - スケール係数を保存して後でバウンディングボックスを元のサイズに変換

2. **推論パラメータの最適化**
   ```python
   results = self.model.predict(
       source=img,
       conf=conf_thres,  # 信頼度閾値
       iou=0.45,         # IoUしきい値（重複検出の除去）
       max_det=100,      # 最大検出数
       device=self.device,
       verbose=False     # 冗長な出力を抑制
   )
   ```
   - 信頼度閾値0.3で低信頼度の誤検出を除外
   - IoUしきい値0.45で適切な重複排除
   - 1画像あたり最大100オブジェクトの検出

3. **重要オブジェクトの優先処理**
   ```python
   max_objects = 50  # 1画像あたりの最大処理オブジェクト数
   if len(boxes) > max_objects:
       # 信頼度でソートして上位のみ処理
       conf_values = boxes.conf.cpu().numpy()
       sorted_indices = np.argsort(-conf_values)[:max_objects]
   ```
   - 最大50オブジェクトに絞り込みによる効率化
   - 信頼度スコアによる優先順位付け

4. **切り抜き画像の保存**
   ```python
   cropped = original_img.crop((x1, y1, x2, y2))
   out_path = os.path.join(
       out_dir, f"{basename_no_ext}_{i}_{class_name}.jpg"
   )
   cropped.save(out_path)
   ```
   - 元の高解像度画像から直接切り抜き
   - ファイル名に元画像名、オブジェクトID、クラス名を含む

### 2.2 CLIP：マルチモーダル特徴抽出

```python
class CLIPFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
```

**実装詳細**：

1. **画像特徴抽出**
   ```python
   def get_image_embedding(self, image_path):
       pil_image = Image.open(image_path).convert("RGB")
       inputs = self.processor(images=pil_image, return_tensors="pt")
       inputs = {k: v.to(self.device) for k, v in inputs.items()}
       with torch.no_grad():
           image_features = self.model.get_image_features(**inputs)
       return image_features.cpu().numpy().squeeze(0)
   ```
   - 画像を開き、RGB形式に変換
   - CLIPプロセッサによる前処理（リサイズ、正規化）
   - 推論時に勾配計算を無効化
   - デバイス間転送を最小化

2. **バッチ処理による高速化**
   ```python
   def get_image_embeddings_batch(self, image_paths, batch_size=32):
       for i in range(0, total, batch_size):
           batch_paths = image_paths[i:i+batch_size]
           # バッチ処理
           batch_images = []
           valid_indices = []
           
           for j, path in enumerate(batch_paths):
               try:
                   img = Image.open(path).convert("RGB")
                   batch_images.append(img)
                   valid_indices.append(j)
               except Exception as e:
                   print(f"画像読み込みエラー {path}: {e}")
           
           if not batch_images:
               continue
           
           inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
           inputs = {k: v.to(self.device) for k, v in inputs.items()}
           
           with torch.no_grad():
               batch_features = self.model.get_image_features(**inputs)
   ```
   - 32画像を同時処理（GPUメモリ効率と処理速度の最適バランス）
   - 無効な画像を自動スキップする堅牢性
   - バッチ内でのパディング最適化

3. **テキスト特徴抽出**
   ```python
   def get_text_embedding(self, text_list):
       inputs = self.processor(text=text_list, return_tensors="pt",
                             padding=True, truncation=True)
       inputs = {k: v.to(self.device) for k, v in inputs.items()}
       with torch.no_grad():
           text_features = self.model.get_text_features(**inputs)
       return text_features.cpu().numpy()
   ```
   - テキストリストを一括処理
   - 自動パディングと切り詰めによる可変長テキスト対応
   - 画像特徴と同じ埋め込み空間への投影

4. **意味的類似度計算**
   ```python
   def parse_text_with_probability(base_text, candidate_labels, clip_extractor):
       text_emb = clip_extractor.get_text_embedding([base_text])
       labels_emb = clip_extractor.get_text_embedding(candidate_labels)

       text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
       labels_emb_norm = labels_emb / np.linalg.norm(labels_emb, axis=1, keepdims=True)

       cos_sims = []
       for i in range(labels_emb_norm.shape[0]):
           sim = (text_emb_norm[0] * labels_emb_norm[i]).sum()
           cos_sims.append(sim)
   ```
   - ベクトル正規化によるコサイン類似度計算
   - スコアのソートと辞書形式での返却
   - 負の類似度の除外による関連概念のみの抽出

### 2.3 Qwen2-VL：画像キャプション生成

```python
def generate_descriptions_batch(image_paths, model, processor, batch_size=8, detected_classes=None):
    # バッチ処理で画像キャプションを生成
```

**実装詳細**：

1. **最適バッチサイズの動的決定**
   ```python
   optimal_batch_size = get_optimal_batch_size(initial_size=batch_size)
   if optimal_batch_size != batch_size:
       print(f"バッチサイズを {batch_size} から {optimal_batch_size} に調整しました（メモリ最適化）")
       batch_size = optimal_batch_size
   ```
   - 利用可能なGPUメモリに基づく自動調整
   - メモリ不足エラーの事前防止

2. **コンテキスト特化型プロンプト設計**
   ```python
   message = [
       {
           "role": "user",
           "content": [
               {
                   "type": "image",
                   "image": f"data:image/jpeg;base64,{base64_image}",
               },
               {
                   "type": "text",
                   "text": "You are an AI vision system specialized in describing images seen from cameras in cars when driving. "
                           "Focus on unusual or dangerous items that may be relevant to driving safety. "
                           "Analyze the following image and list up to 10 notable objects/scenarios."
                           f"{class_hint}"
                           f"Note that max tokens is 150."
               },
           ],
       }
   ]
   ```
   - 運転安全性という特定コンテキストに特化
   - YOLOの検出結果を追加情報として提供
   - トークン制限による簡潔な出力の促進

3. **混合精度推論の活用**
   ```python
   with torch.cuda.amp.autocast(enabled=True):  # 混合精度を有効化
       generated_ids = model.generate(
           **inputs, 
           max_new_tokens=150,
           do_sample=False,
           num_beams=1,
           temperature=1.0
       )
   ```
   - FP16演算による高速化とメモリ使用量削減
   - 確定的な生成（do_sample=False）による再現性確保
   - 生成トークン数の制限による効率化

4. **サブバッチ分割処理**
   ```python
   sub_batch_size = 100  # 一度に処理する画像数
   all_descriptions = []
   
   for i in range(0, len(process_paths), sub_batch_size):
       sub_batch_paths = process_paths[i:i+sub_batch_size]
       sub_batch_indices = process_indices[i:min(i+sub_batch_size, len(process_indices))]
       
       # サブバッチのキャプション生成
       sub_batch_descriptions = generate_descriptions_batch(...)
       
       all_descriptions.extend(sub_batch_descriptions)
       
       # サブバッチ処理後にGPUメモリを解放
       torch.cuda.empty_cache()
   ```
   - 大規模データセット（数千画像）の分割処理
   - 各サブバッチ後のメモリ解放による長時間安定処理

## 3. 異常検知技術の実装詳細

### 3.1 Isolation Forest：特徴空間での異常検知

```python
def train_isolation_forest(features, contamination=0.1):
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(features)
    return iso

def predict_outliers(features, iso_model):
    if len(features) == 0:
        return []
    return iso_model.predict(features)
```

**実装詳細**：

1. **パラメータ設計**
   - contamination=0.1：データセット中の異常値の予想割合（10%）
   - random_state=42：再現性のための固定シード値

2. **CLIPの埋め込み空間での効率的な動作**
   - 512次元の高次元空間での分離性能の高さ
   - ランダム分割による計算効率の良さ

3. **二値化された判定**
   - 1: インライア（正常）
   - -1: アウトライア（異常）

### 3.2 t-SNE：高次元特徴の可視化

```python
def visualize_tsne(features, labels=None, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, random_state=42,
               perplexity=min(30, max(5, len(features) - 1)))
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    if labels is not None:
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=labels, cmap="tab10", alpha=0.7
        )
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
```

**実装詳細**：

1. **パラメータの動的調整**
   - perplexity値の自動調整：データ数に応じて5〜30の範囲で設定
   - 少数サンプルでも安定した可視化を実現

2. **ラベル情報の活用**
   - クラス別の色分け表示によるパターン認識の容易化
   - クラスごとの分布の可視化

3. **可視化の保存と出力**
   - 標準化されたファイル名での保存
   - 高解像度での出力（10×7インチ）

### 3.3 概念ベースの意味的異常検知

```python
# キャプション解析
top_concept = None
is_final_outlier = False

if desc:
    # キャプション判定ロジック
    desc_lower = desc.lower()
    is_failed_caption = False
    
    if any(phrase in desc_lower for phrase in ["i'm sorry", "unable to", "i'm unable to", "cannot process"]):
        is_failed_caption = True
    
    if not is_failed_caption:
        # 類似度計算
        probs_dict = parse_text_with_probability(desc, concept_list, clip_extractor)
        
        if probs_dict:
            top_concept = next(iter(probs_dict))
            # final outlier の判定基準: top_concept が common_classes にないこと
            if top_concept not in common_classes:
                is_final_outlier = True
```

**実装詳細**：

1. **2段階の異常検知**
   - 特徴空間での異常（Isolation Forest）
   - 意味的概念に基づく異常（common_classesに含まれないか）

2. **エラー処理**
   - キャプション生成失敗の検出と処理
   - エラーメッセージに基づく失敗判定

3. **意味的類似度の活用**
   - 生成されたキャプションと概念リストとの類似度計算
   - 最も類似度の高い概念の抽出と判定への活用

## 4. モデル統合によるシステム最適化

### 4.1 マルチモダリティの統合

```python
# YOLOの検出結果をQwen2-VLに提供
detected_class = detected_classes.get(path, "") if detected_classes else ""
class_hint = f" Note that YOLO says the image is about a {detected_class}." if detected_class else ""

# キャプションとCLIPの概念マッチング
probs_dict = parse_text_with_probability(desc, concept_list, clip_extractor)
```

**実装詳細**：

1. **情報の相互活用**
   - YOLOの検出結果をQwen2-VLのプロンプトに統合
   - Qwen2-VLのキャプションをCLIPで分析

2. **段階的処理と情報の流れ**
   - 物体検出 → 特徴抽出 → キャプション生成 → 概念マッチング
   - 各段階の出力を次の段階に活用する設計

### 4.2 計算効率の最適化

```python
# サブバッチ処理と進捗表示
for i in range(0, len(process_paths), sub_batch_size):
    # ... 処理 ...
    
    # 進捗表示
    elapsed = time.time() - start_time
    progress = (i + 1) / total_images * 100
    eta = (elapsed / (i + 1)) * (total_images - (i + 1)) if i > 0 else 0
    print(f"キャプション生成: {progress:.1f}% ({i+1}/{total_images}) | 経過: {elapsed:.1f}秒 | 残り: {eta:.1f}秒")
```

**実装詳細**：

1. **メモリ使用の最適化**
   - 段階的処理による一度に必要なメモリの削減
   - 明示的なGPUメモリ解放

2. **進捗追跡と時間予測**
   - 詳細な進捗表示による長時間処理の監視
   - 残り時間予測による処理計画の支援

## 5. モデル選択とパラメータチューニング

### 5.1 YOLOモデルの選択基準

- **YOLOv8n (Nano)**
  - 最小サイズモデル（3.2M）
  - 高速だが精度は低め
  - リソースが極めて限られた環境向け

- **YOLOv8s (Small)**
  - 小サイズモデル（11.2M）
  - 速度と精度のバランスが良い
  - モバイルデバイスなどのリソース制約環境向け

- **YOLOv8m (Medium)**
  - 中サイズモデル（25.9M）
  - 一般的な用途に適したバランス
  - 標準的なGPU環境向け

- **YOLOv8l (Large)** ← 本システムでの選択
  - 大サイズモデル（43.7M）
  - 高精度だが処理速度はやや低下
  - 検出精度が重要な応用向け

- **YOLOv8x (XLarge)**
  - 超大型モデル（68.2M）
  - 最高精度だが処理速度は最も低い
  - 研究用途や高精度が必須の場合向け

### 5.2 Qwen2-VLモデルのサイズ選択

- **Qwen2-VL-2B** ← 標準選択
  - パラメータ数：2B（20億）
  - 特徴：
    - 推論速度が速い（1画像あたり約0.5〜1秒）
    - 8GB以上のGPUメモリで動作可能
    - 基本的な物体認識と説明生成が可能

- **Qwen2-VL-7B** ← 高精度オプション
  - パラメータ数：7B（70億）
  - 特徴：
    - より高度なシーン理解と詳細な説明が可能
    - 複雑なオブジェクトや状況の認識精度が向上
    - 16GB以上のGPUメモリが必要
    - 推論速度はやや遅い（1画像あたり約1〜2秒）

### 5.3 主要パラメータの最適値

| パラメータ | 標準値 | 推奨範囲 | 影響 |
|------------|--------|----------|------|
| conf_thres (YOLO) | 0.3 | 0.1 - 0.5 | 検出感度。低いと多くのオブジェクトを検出するが誤検出も増加 |
| contamination (IsoForest) | 0.1 | 0.01 - 0.2 | 想定される異常値の割合。高いと多くの異常を検出するが誤検出も増加 |
| batch_size | 32 (CLIP), 8 (Qwen2-VL) | GPU依存 | 大きいほど処理は速いがメモリ使用量が増加 |
| max_new_tokens | 150 | 50 - 300 | 生成されるキャプションの最大長。長いほど詳細な説明が可能だがメモリと時間が増加 |
| threshold_percentile | 95 | 90 - 99 | 異常スコアの閾値パーセンタイル。高いほど厳格な異常判定 |

## 6. モデルの限界と改善方向

### 6.1 現在の限界

- **YOLOの検出限界**：
  - 学習データに存在しない珍しい物体の検出精度が低い
  - 物体の部分的な遮蔽への対応が不十分
  - 小さい物体の検出精度が低い

- **CLIPの表現限界**：
  - 細かな視覚的違いの区別が不十分
  - ドメイン特化型の認識タスクでの精度限界
  - 新奇な概念や状況の理解が制限的

- **Qwen2-VLの生成限界**：
  - 150トークン制限による説明の簡略化
  - 稀少物体に対する説明生成精度の不安定さ
  - 推論時間とリソース要求の高さ

### 6.2 改善の方向性

- **モデルの拡張**：
  - 特定ドメイン（例：交通シーン）に特化した微調整
  - 複数の物体検出器の結果を統合するアンサンブル手法
  - より大規模なビジョン言語モデルの導入（例：GPT-4V）

- **アーキテクチャの改善**：
  - 物体関係の理解を強化するグラフニューラルネットワークの統合
  - 時間的文脈を考慮したシーケンシャルモデルの導入
  - セグメンテーションモデルによる詳細な物体境界抽出

- **学習とチューニング**：
  - ドメイン適応技術による特定環境への最適化
  - 異常検知アルゴリズムの洗練（例：Deep SVDDの導入）
  - より大規模で多様なデータセットでの事前学習 