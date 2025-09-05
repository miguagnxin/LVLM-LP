# 文件: model/internvl3.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from model.base import LargeMultimodalModel


class InternVL3(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        model_name = args.model_path or "OpenGVLab/InternVL3-1B-hf"

        # 加载 tokenizer / processor / model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        # 推理参数
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = getattr(args, "max_new_tokens", 100)

    def forward_with_probs(self, image, prompt):
        """
        image: numpy array (RGB)，来自 run_model 里的 cv2.imread + cvtColor
        prompt: str
        """

        # 1. 组装 messages 格式
        messages = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]

        # 2. processor 处理
        inputs = self.processor(
            text=messages,
            return_tensors="pt"
        ).to(self.model.device)

        # 3. 调用 generate，开启 output_scores 以拿到 logits/probs
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True
            )

        # 4. 提取 response
        output_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 5. 提取 logits & probs
        logits = torch.stack(outputs.scores, dim=0).cpu().numpy()
        probs = torch.nn.functional.softmax(torch.stack(outputs.scores, dim=0), dim=-1).cpu().numpy()

        # 转换 output_ids -> numpy，保证和 run_model 兼容
        return response, output_ids.cpu().numpy(), logits, probs
