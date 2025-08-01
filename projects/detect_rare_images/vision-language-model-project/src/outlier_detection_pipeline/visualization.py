#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import numpy as np

def visualize_tsne(features, labels=None, title="t-SNE Visualization", out_dir="out_images"):
    """
    t-SNEを使用して特徴の可視化を行う関数
    
    Args:
        features: shape=(N, D)
        labels:   shape=(N,) or None
        title:    図のタイトル
        out_dir:  出力ディレクトリのパス
    """
    if len(features) == 0:
        print("No features to visualize.")
        return

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
    plt.title(title)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    save_name = title.lower().replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    out_tsne_path = os.path.join(out_dir, f"{save_name}.png")
    plt.savefig(out_tsne_path)
    plt.close()
    return out_tsne_path

def visualize_class_tsne(features, class_names, title="t-SNE Visualization by Class", top_n=20, out_dir="out_images"):
    """
    クラス別にt-SNE可視化を行う関数
    
    Args:
        features: shape=(N, D)
        class_names: クラス名のリスト
        title: 図のタイトル
        top_n: 表示する上位クラス数
        out_dir: 出力ディレクトリのパス
    """
    if len(features) == 0 or len(class_names) == 0:
        print("No features or class names to visualize.")
        return

    # t-SNE次元削減
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(features) - 1)))
    tsne_result = tsne.fit_transform(features)
    
    # クラスの出現回数をカウント
    from collections import Counter
    class_counts = Counter(class_names)
    top_classes = [cls for cls, count in class_counts.most_common(top_n)]
    
    # 可視化
    plt.figure(figsize=(15, 10))
    cmap = plt.get_cmap("tab20")
    
    # クラス別に色分けして表示
    for i, cls in enumerate(top_classes):
        indices = [j for j, name in enumerate(class_names) if name == cls]
        if indices:
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                       label=f"{cls} ({len(indices)})", 
                       color=cmap(i/len(top_classes)), 
                       alpha=0.7)
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    return out_path, tsne_result

def visualize_outliers_tsne(tsne_result, outlier_flags, method_name="Combined", title=None, out_dir="out_images"):
    """
    アウトライア検出結果をt-SNEで可視化する関数
    
    Args:
        tsne_result: t-SNEで次元削減された特徴量 (N, 2)
        outlier_flags: アウトライアフラグ (1: アウトライア, 0: インライア)
        method_name: 検出手法の名前
        title: 図のタイトル（Noneの場合は自動生成）
        out_dir: 出力ディレクトリのパス
    
    Returns:
        保存されたファイルのパス
    """
    if title is None:
        title = f"t-SNE with {method_name} Outliers"
    
    plt.figure(figsize=(12, 8))
    
    # インライアとアウトライアに分ける
    inliers = tsne_result[outlier_flags == 0]
    outliers = tsne_result[outlier_flags == 1]
    
    # インライアを青、アウトライアを赤で表示
    plt.scatter(inliers[:, 0], inliers[:, 1], c='lightblue', alpha=0.7, label='Inliers')
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', alpha=0.7, label='Outliers')
    
    # 凡例とタイトル
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    return out_path

def visualize_multiple_outliers_tsne(tsne_result, outlier_flags_dict, title="t-SNE with Multiple Outlier Methods", out_dir="out_images"):
    """
    複数の手法によるアウトライア検出結果を同じt-SNEプロットで可視化する関数
    
    Args:
        tsne_result: t-SNEで次元削減された特徴量 (N, 2)
        outlier_flags_dict: 手法名からアウトライアフラグへの辞書 {手法名: outlier_flags}
        title: 図のタイトル
        out_dir: 出力ディレクトリのパス
    
    Returns:
        保存されたファイルのパス
    """
    plt.figure(figsize=(15, 10))
    
    # サブプロットの数と配置を決定
    n_methods = len(outlier_flags_dict)
    n_rows = (n_methods + 1) // 2  # 2列で並べる
    n_cols = min(2, n_methods)
    
    for i, (method_name, flags) in enumerate(outlier_flags_dict.items(), 1):
        ax = plt.subplot(n_rows, n_cols, i)
        
        # インライアとアウトライアに分ける
        inliers = tsne_result[flags == 0]
        outliers = tsne_result[flags == 1]
        
        # インライアを青、アウトライアを赤で表示
        ax.scatter(inliers[:, 0], inliers[:, 1], c='lightblue', alpha=0.7, label='Inliers')
        ax.scatter(outliers[:, 0], outliers[:, 1], c='red', alpha=0.7, label='Outliers')
        
        # 凡例とタイトル
        ax.set_title(f"{method_name} (Outliers: {len(outliers)})")
        ax.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    return out_path

def plot_concept_probabilities(probs_dict, out_path=None, top_n=10, title="Top Concept Similarities"):
    """
    コンセプト類似度のグラフを作成する関数
    
    Args:
        probs_dict: コンセプト名から確率への辞書
        out_path: 出力ファイルパス（Noneの場合は表示のみ）
        top_n: 表示する上位の数
        title: グラフのタイトル
    """
    plt.figure(figsize=(8, 4))
    sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)[:top_n]
    sorted_probs_keys = [k for k, v in sorted_probs]
    sorted_probs_values = [v for k, v in sorted_probs]
    
    plt.bar(sorted_probs_keys, sorted_probs_values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path
    else:
        plt.show()
        plt.close()
        return None 