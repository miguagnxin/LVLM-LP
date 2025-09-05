import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from model.base import LargeMultimodalModel


class InternVL3(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        model_name = args.model_path or "OpenGVLab/InternVL3-1B-hf"

        # 加载 tokenizer / processor / model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()

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

        # 2. 用 processor 处理多模态输入（必须同时传 messages 和 images）
        inputs = self.processor(
            prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # 3. generate 推理
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

        # 4. 提取 response（去掉输入 tokens）
        input_len = inputs["input_ids"].shape[1]
        output_ids = outputs.sequences[0][input_len:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 5. 提取 logits & probs
        if outputs.scores is not None:
            scores = torch.stack(outputs.scores, dim=0)  # [seq_len, batch, vocab]
            logits = scores.squeeze(1).cpu().numpy()     # [seq_len, vocab]
            probs = torch.nn.functional.softmax(scores, dim=-1).squeeze(1).cpu().numpy()
        else:
            logits, probs = None, None

        return response, output_ids.cpu().numpy(), logits, probs
