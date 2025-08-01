#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from PIL import Image
import numpy as np
from .utils import get_device

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not found. Install via 'pip install ultralytics'.")

class DetectorYOLO:
    """
    YOLOv8ベースのオブジェクト検出クラス。
    """
    def __init__(self, model_path="yolov8l.pt", device=None):
        if device is None:
            device = get_device()
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_and_crop(self, image_path, out_dir="cropped_objects", conf_thres=0.25):
        """
        画像内のオブジェクトを検出し、信頼度が閾値を超えるオブジェクトを切り抜いて保存します。

        Returns:
            cropped_info_list: [(out_path, class_name, conf, xyxy, image_path), ...]
        """
        os.makedirs(out_dir, exist_ok=True)
        try:
            # 画像が存在するか確認
            if not os.path.isfile(image_path):
                print(f"画像ファイルが見つかりません: {image_path}")
                return []
            
            original_img = Image.open(image_path).convert("RGB")
            original_width, original_height = original_img.size
            
            # 処理用に画像をリサイズ
            img = original_img.copy()
            width, height = img.size
            max_size = 1280
            scale_factor = 1.0
            
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                    scale_factor = original_width / new_width
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                    scale_factor = original_height / new_height
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # バッチ処理モードを使用
            results = self.model.predict(
                source=img,
                conf=conf_thres,
                iou=0.45,
                max_det=100,
                device=self.device,
                verbose=False  # 冗長な出力を抑制
            )
            
            if len(results) == 0:
                return []

            boxes = results[0].boxes
            names = self.model.names
            cropped_info_list = []

            # 検出結果が多すぎる場合は上位のみ処理
            max_objects = 50  # 1画像あたりの最大処理オブジェクト数
            num_boxes = min(len(boxes), max_objects)
            
            if len(boxes) > max_objects:
                # 信頼度でソートして上位のみ処理
                conf_values = boxes.conf.cpu().numpy()
                sorted_indices = np.argsort(-conf_values)[:max_objects]
            else:
                sorted_indices = range(len(boxes))

            for i in sorted_indices:
                # リサイズされた画像での座標
                xyxy_resized = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = names[cls_id] if cls_id in names else f"class_{cls_id}"

                # 元の解像度に座標を戻す
                if scale_factor != 1.0:
                    x1, y1, x2, y2 = xyxy_resized
                    x1_orig = int(x1 * scale_factor)
                    y1_orig = int(y1 * scale_factor)
                    x2_orig = int(x2 * scale_factor)
                    y2_orig = int(y2 * scale_factor)
                else:
                    x1, y1, x2, y2 = xyxy_resized
                    x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2

                # 検出領域を拡大（一回り大きく）
                width = x2_orig - x1_orig
                height = y2_orig - y1_orig
                padding_x = int(width * 0.05)  # 横方向に20%拡大
                padding_y = int(height * 0.05)  # 縦方向に20%拡大

                x1_expanded = max(0, x1_orig - padding_x)
                y1_expanded = max(0, y1_orig - padding_y)
                x2_expanded = min(original_width, x2_orig + padding_x)
                y2_expanded = min(original_height, y2_orig + padding_y)

                # 拡大したバウンディングボックスを使用
                xyxy = (x1_expanded, y1_expanded, x2_expanded, y2_expanded)

                # 元の解像度の画像から切り抜く
                x1, y1, x2, y2 = xyxy
                cropped = original_img.crop((x1, y1, x2, y2))

                basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(
                    out_dir, f"{basename_no_ext}_{i}_{class_name}.jpg"
                )
                cropped.save(out_path)
                cropped_info_list.append((out_path, class_name, conf, xyxy, image_path))

            return cropped_info_list
        except Exception as e:
            print(f"画像処理エラー {image_path}: {e}")
            return [] 