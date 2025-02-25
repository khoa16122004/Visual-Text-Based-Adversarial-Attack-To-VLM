import torch
import clip
from typing import List
import torch.nn as nn
from PIL import Image
# from transformers import CLIPTokenizer

class CLIP:
    def __init__(self, model_name="ViT-B/16"):
        self.model, self.preprocess = clip.load(model_name, device="cuda")
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model.eval()
        
    @torch.no_grad()
    def text_encode(self, c: List[str]):
        c = self.tokenizer(c).cuda()
        return self.model.encode_text(c)
    
    @torch.no_grad()
    def image_encode(self, img: List[Image.Image]):
        img = self.preprocess(img).cuda()
        return self.model.encode_image(img)
    def evaluate(self, x: Image, c: str):
        # minimize cos(f(x), g(c))
        x = self.preprocess(x).cuda()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
                    
        c = clip.tokenize(c).cuda()
        if len(c.shape) == 3:
            c = c.unsqueeze(0)
            
        with torch.no_grad():
            img_features = self.model.encode_image(x)
            text_features = self.model.encode_text(c)
        
        cos_sim = self.cos(img_features, text_features)
        
        return cos_sim
if __name__ == "__main__":
    model = CLIP(model_name="ViT-B/16")
    img_path = r'D:\codePJ\RESEARCH\Flow-Based-Attack-To-VLM\src\images\lionsea.jpg'
    img = Image.open(img_path)
    text = 'a lion in the jungle'
    try:
        score = model.evaluate(img, text)
        print(f"TEST MODEL: {score}")
    except Exception as e:
        print(f"Error: {e}")