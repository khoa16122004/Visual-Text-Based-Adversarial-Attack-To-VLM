# import clip
import torch
import torch.nn as nn
from PIL import Image
# import open_clip
from typing import List
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

class OpenCLIP:
    def __init__(self, model_name="ViT-H-14"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2B-s32B-b79K')
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model.eval().cuda()
    
    @torch.no_grad()
    def text_encode(self, c: List[str]):
        c = self.tokenizer(c).cuda()
        return self.model.encode_text(c)
    
    @torch.no_grad()
    def image_encode(self, imgs: List[Image.Image]):
        img_torch = torch.stack([self.preprocess(img) for img in imgs]).cuda()
        return self.model.encode_image(img_torch)
    
        
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
        
        img_features = img_features / torch.norm(img_features, dim=1)
        
        cos_sim = self.cos(img_features, text_features)
        
        return cos_sim
    
class CLIP:
    def __init__(self, model_name="ViT-L/14"):
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

class BLIP:
    def __init__(self):
        self.model, self.vis_proccessors, self.text_proccessors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device="cuda", is_eval=True)
        print(self.model)
        
        
    @torch.no_grad()
    def text_encode(self, c: List[str]):
        pass
    
    @torch.no_grad()
    def image_encode(self, img: List[Image.Image]):
        pass
    
    def evaluate(self, x: Image, c: str):
        pass

     
if __name__ == "__main__":
    # model = CLIP()
    # model = OpenCLIP()
    model = BLIP()