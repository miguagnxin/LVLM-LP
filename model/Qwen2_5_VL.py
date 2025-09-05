import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from model.base import LargeMultimodalModel


class Qwen2_5_VL(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        model_name = args.model_path or "Qwen/Qwen2.5-VL-3B-Instruct"

        # 加载 tokenizer / processor / model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        # 推理参数
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = getattr(args, "max_new_tokens", 100)

    def forward_with_probs(self, image, prompt):
        """
        image: numpy array (RGB)，来自 run_model.py 的 cv2.imread + cvtColor
        prompt: str
        """

        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 使用processor处理多模态输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)

        # 生成推理
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

        # 提取response（去掉输入tokens）
        input_len = inputs["input_ids"].shape[1]
        output_ids = outputs.sequences[0][input_len:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 提取logits & probs
        if outputs.scores is not None:
            scores = torch.stack(outputs.scores, dim=0)  # [seq_len, batch, vocab]
            logits = scores.squeeze(1).cpu().numpy()     # [seq_len, vocab]
            probs = torch.nn.functional.softmax(scores, dim=-1).squeeze(1).cpu().numpy()
        else:
            logits, probs = None, None

        return response, output_ids.cpu().numpy(), logits, probs
