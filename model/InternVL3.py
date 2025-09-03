import sys
import os
import json

import torch
from torch import nn
import numpy as np
from io import BytesIO
from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput

from model.base import LargeMultimodalModel

class InternVL3(LargeMultimodalModel):
    def __init__(self, args):
        super(InternVL3, self).__init__()
        
        # Import InternVL3 components
        try:
            from transformers import AutoTokenizer, AutoModel
            from PIL import Image
            import torchvision.transforms as transforms
        except ImportError as e:
            print(f"Required packages not installed: {e}")
            print("Please install: pip install transformers torch torchvision pillow")
            sys.exit(1)
        
        # Load model and tokenizer
        self.model_path = args.model_path
        print(f"Loading InternVL3 model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        ).eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        
        print("InternVL3 model loaded successfully")
    
    def refresh_chat(self):
        """Reset conversation state"""
        self.conv = []
    
    def _basic_forward(self, image, prompt, return_dict=False):
        """Basic forward pass with InternVL3"""
        self.refresh_chat()
        
        # Prepare image
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        # Prepare input
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prepare text input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate response
        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=100,
                use_cache=True,
                return_dict_in_generate=return_dict,
                output_attentions=return_dict,
                output_hidden_states=return_dict,
                output_scores=return_dict
            )
        
        return inputs.input_ids, outputs
    
    def forward_with_probs(self, image, prompt):
        """Forward pass that returns probabilities and logits"""
        input_ids, outputs = self._basic_forward(image, prompt, return_dict=True)
        
        if isinstance(outputs, BeamSearchDecoderOnlyOutput):
            beam_indices = outputs['beam_indices'][0].cpu()
            beam_indices = [i for i in beam_indices if i != -1]
            logits = None
            probs = float(outputs['sequences_scores'].cpu().item())
            output_ids = outputs["sequences"][0][-len(beam_indices):]
        else:
            # Extract logits and probabilities
            if hasattr(outputs, 'scores') and outputs.scores:
                logits = torch.cat(outputs.scores, dim=0).cpu().numpy()
                probs = [nn.functional.softmax(next_token_scores, dim=-1) 
                        for next_token_scores in outputs.scores]
                probs = torch.cat(probs).cpu().numpy()
            else:
                logits = None
                probs = None
            
            output_ids = outputs.sequences[0][len(input_ids):]
        
        # Decode response
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Convert to numpy
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().numpy()
        
        return response, output_ids, logits, probs
    
    def forward(self, image, prompt):
        """Simple forward pass that returns only the response"""
        input_ids, outputs = self._basic_forward(image, prompt, return_dict=False)
        
        # Decode response
        if hasattr(outputs, 'sequences'):
            output_ids = outputs.sequences[0][len(input_ids):]
        else:
            output_ids = outputs[0][len(input_ids):]
        
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return response
