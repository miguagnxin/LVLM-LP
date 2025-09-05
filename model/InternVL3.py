# 文件：model/internvl3.py

from model.base import LargeMultimodalModel  # 根据你的项目结构
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

class InternVL3(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        model_name = args.model_path or "OpenGVLab/InternVL3-1B-hf"
        # 加载 pipeline 简化推理逻辑
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            device=0  # 根据需求调整 GPU 设备 id
        )
        self.max_new_tokens = getattr(args, "max_new_tokens", 50)
        self.return_full_text = getattr(args, "return_full_text", False)

    def forward_with_probs(self, image, prompt):
        # 构造消息格式
        messages = [
            {"type": "image", "image": image},  # image path or PIL.Image
            {"type": "text", "text": prompt}
        ]
        outputs = self.pipe(text=messages,
                            max_new_tokens=self.max_new_tokens,
                            return_full_text=self.return_full_text)
        generated = outputs[0]["generated_text"]
        # 该 pipeline 输出不包含 token 概率等，此处可返回 None
        return generated, None, None, None
