import clip
import torch
from PIL import Image

class CLIPModel():
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.model.eval()

    def get_img_embedding(self, img_path):

        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.model.encode_image(img)
        
        emb /= emb.norm(dim=1, keepdim=True)

        return emb.cpu().numpy()
    
    def get_txt_embedding(self, query):

        query = clip.tokenize(query)
        query = query.to(self.device)
        
        with torch.no_grad():
            emb = self.model.encode_text(query)
        
        emb /= emb.norm(dim=-1, keepdim=True)
        
        return emb.cpu().numpy()