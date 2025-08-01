#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def encode_image(image_path):
    """指定パスの画像をbase64エンコードする"""
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def generate_description(image_path, model, processor):
    """LLaVAを用いて画像のキャプションを生成する関数"""
    base64_image = encode_image(image_path)
    if not base64_image:
        print("画像のエンコードに失敗しました。")
        return None
                
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI vision system specialized in describing images. "
                "Focus on unusual or domain-relevant items."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the following image and list up to 10 notable objects/scenarios."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    try:
        # apply_chat_template により、入力用のテンソルが作成される
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": base64_image}},
                {"type": "text", "text": "Analyze this image and list up to 10 notable objects/scenarios."},
            ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation, #messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        # テンソルのみデバイス変換を実施
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor):
                        v[sub_k] = sub_v.to(model.device)
                inputs[k] = v

        # モデルからキャプションを生成
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        response = processor.batch_decode(generate_ids, skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"LLaVA model error: {e}")
        return None

def main():
    # ※必ず存在するテスト画像のパスに変更すること
    image_path = "projects/detect_rare_images/data/test_data/test_7.png"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # LLaVAモデルとprocessorの読み込み（モデルファイルのパス・バージョンは環境に合わせる）
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    response = generate_description(image_path, model, processor)
    print("LLaVA response:", response)

if __name__ == "__main__":
    main()
