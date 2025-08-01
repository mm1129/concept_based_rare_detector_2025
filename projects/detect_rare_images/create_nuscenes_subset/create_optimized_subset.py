#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import shutil
import numpy as np
from collections import Counter, defaultdict

def create_optimized_subset(
    detection_results_path,
    output_dir,
    second_results_path=None,
    base_random_ratio=0.1,
    target_ratio=0.1,
    target_classes=None,
    seed=42
):
    """
    検出結果から最適化されたサブセットを作成する関数
    
    Args:
        detection_results_path: 検出結果のJSONファイルパス
        output_dir: 出力ディレクトリ
        second_results_path: 2つ目の検出結果のJSONファイルパス (オプション)
        base_random_ratio: ベースとなるランダムサンプルの割合
        target_ratio: ターゲットサンプルの割合
        target_classes: ターゲットクラスのリスト
        seed: 乱数シード
    """
    # シード値を設定
    random.seed(seed)
    np.random.seed(seed)
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 検出結果を読み込む
    with open(detection_results_path, 'r') as f:
        detection_results = json.load(f)
    
    # 2つ目の検出結果がある場合は合体
    if second_results_path:
        with open(second_results_path, 'r') as f:
            second_results = json.load(f)
        detection_results = detection_results + second_results
    # サンプルトークンごとに結果をグループ化
    samples_by_token = defaultdict(list)
    for result in detection_results:
        sample_token = result.get('sample_token')
        if sample_token:
            samples_by_token[sample_token].append(result)
    
    # 全サンプルトークンのリスト
    all_tokens = list(samples_by_token.keys())
    total_samples = len(all_tokens)
    
    # ベースとなるランダムサンプルを選択
    base_random_count = int(total_samples * base_random_ratio)
    base_random_tokens = set(random.sample(all_tokens, base_random_count))
    
    # ターゲットクラスを含むサンプルを選択
    target_samples = []
    for token, results in samples_by_token.items():
        if token in base_random_tokens:
            continue
            
        is_target = False
        is_final_outlier = False
        
        for result in results:
            # ターゲットクラスチェックを改善
            if target_classes and result.get('top_concept'):
                top_concept = result.get('top_concept').lower().strip()
                # スペルミスの可能性がある文字列を正規化
                top_concept = top_concept.replace('vechicle', 'vehicle')
                
                for target in target_classes:
                    if target.lower() in top_concept:
                        is_target = True
                        break
            
            # 最終アウトライアチェック
            if result.get('is_final_outlier'):
                is_final_outlier = True
        
        # ターゲットクラスまたは最終アウトライアの場合、候補に追加
        if is_target or is_final_outlier:
            target_samples.append((token, is_target, is_final_outlier))
    
    # ターゲットサンプルをシャッフルして選択
    random.shuffle(target_samples)
    target_count = int(total_samples * target_ratio)
    target_count = min(target_count, len(target_samples))
    
    # ターゲットクラスと最終アウトライアを優先
    selected_target_tokens = set()
    
    # 1. ターゲットクラスかつ最終アウトライアを優先
    for token, is_target, is_final_outlier in target_samples:
        if is_target and is_final_outlier:
            selected_target_tokens.add(token)
            if len(selected_target_tokens) >= target_count:
                break
    
    # 2. ターゲットクラスを優先
    if len(selected_target_tokens) < target_count:
        for token, is_target, is_final_outlier in target_samples:
            if is_target and token not in selected_target_tokens:
                selected_target_tokens.add(token)
                if len(selected_target_tokens) >= target_count:
                    break
    
    # 3. 最終アウトライアを優先
    if len(selected_target_tokens) < target_count:
        for token, is_target, is_final_outlier in target_samples:
            if is_final_outlier and token not in selected_target_tokens:
                selected_target_tokens.add(token)
                if len(selected_target_tokens) >= target_count:
                    break
    
    # 4. 残りをランダムに選択
    remaining_target_tokens = [token for token, _, _ in target_samples 
                              if token not in selected_target_tokens]
    remaining_count = target_count - len(selected_target_tokens)
    
    if remaining_count > 0 and remaining_target_tokens:
        additional_tokens = random.sample(remaining_target_tokens, 
                                         min(remaining_count, len(remaining_target_tokens)))
        selected_target_tokens.update(additional_tokens)
    
    # 最終的な選択サンプル
    final_selected_tokens = base_random_tokens.union(selected_target_tokens)
    
    # 結果を保存する前にログ出力を追加
    concept_counts = Counter()
    for token in final_selected_tokens:
        for result in samples_by_token[token]:
            if result.get('top_concept'):
                concept_counts[result.get('top_concept').lower().strip()] += 1
    
    print("\nSelected concepts distribution:")
    for concept, count in concept_counts.most_common():
        print(f"- {concept}: {count}")
    
    # 結果を保存
    result_info = {
        "total_samples": total_samples,
        "base_random_count": len(base_random_tokens),
        "target_count": len(selected_target_tokens),
        "final_selected_count": len(final_selected_tokens),
        "selected_tokens": list(final_selected_tokens)
    }
    
    result_path = os.path.join(output_dir, "optimized_subset_info.json")
    with open(result_path, 'w') as f:
        json.dump(result_info, f, indent=2)
    
    # 選択されたトークンのリストを保存
    tokens_path = os.path.join(output_dir, "selected_tokens.txt")
    with open(tokens_path, 'w') as f:
        for token in final_selected_tokens:
            f.write(f"{token}\n")
    
    print(f"最適化されたサブセットを作成しました: {len(final_selected_tokens)}サンプル")
    print(f"- ベースランダムサンプル: {len(base_random_tokens)}サンプル")
    print(f"- ターゲットサンプル: {len(selected_target_tokens)}サンプル")
    print(f"結果は {output_dir} に保存されました")
    
    return result_info

def main():
    parser = argparse.ArgumentParser(description="検出結果から最適化されたサブセットを作成するスクリプト")
    parser.add_argument("--detection_results", type=str, default="outlier_detection_results/outlier_detection_0424_13-59-49_JST/process_checkpoint.json",
                        help="検出結果のJSONファイルパス")
    
    parser.add_argument("--output_dir", type=str, default="optimized_subset",
                        help="出力ディレクトリ")
    parser.add_argument("--base_random_ratio", type=float, default=0.1,
                        help="ベースとなるランダムサンプルの割合")
    parser.add_argument("--target_ratio", type=float, default=0.1,
                        help="ターゲットサンプルの割合")
    parser.add_argument("--target_classes", nargs='+', 
                        default=["construction_vehicle", "bicycle", "pedestrian_with_bicycle"],#, "motorcycle", "trailer"
                        help="ターゲットクラスのリスト")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード")
    
    parser.add_argument("--second_results", type=str, default=None,
                        help="2つ目の検出結果のJSONファイルパス（オプション）")
    
    args = parser.parse_args()
    # 入力ファイル名から日時を抽出して出力ディレクトリ名を生成
    # base_name = os.path.basename(args.detection_results)
    # date_str = base_name.split('_')[2]  # outlier_detection_0424_13-59-49_JST から 0424 を抽出
    
    # if args.second_results:
    #     second_base = os.path.basename(args.second_results)
    #     second_date = second_base.split('_')[2]
    #     args.output_dir = f"optimized_subset_{date_str}_{second_date}"
    # else:
    #     args.output_dir = f"optimized_subset_{date_str}"
    
    create_optimized_subset(
        detection_results_path=args.detection_results,
        output_dir=args.output_dir,
        second_results_path=args.second_results,
        base_random_ratio=args.base_random_ratio,
        target_ratio=args.target_ratio,
        target_classes=args.target_classes,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 