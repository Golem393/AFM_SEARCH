"""Finetuned DistilBERT model to extract keywords from prompts

"""
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch

class KeyWordExtractor():
    def __init__(self, tokenizer_path: str, model_path: str):
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    
    def extract(self, input, device):
        
        if torch.cuda.is_available() and device.startswith('cuda'):
            device = torch.device(device)
        elif torch.mps.is_available() and device == 'mps':
            device = torch.device(device)
        else:
            device = torch.device('cpu')
        print(f"[KW-Extractor] device set to {device}")

        model = self.model.to(device)
        model.eval()
        tokenizer = self.tokenizer

        input_tokenized = tokenizer(input, 
                          truncation=True,
                          padding="max_length", 
                          max_length=32, 
                          return_tensors='pt',
                          add_special_tokens=False)
        
        input_token = input_tokenized['input_ids'].to(device)
        attn_mask = input_tokenized['attention_mask'].to(device)

        with torch.no_grad():
            prediction = model(input_token, attn_mask)

        pred_labels = prediction.logits.argmax(dim=-1)
        masked_pred_labels = pred_labels * attn_mask

        # group consecutive token_ids where masked_pred_labels == 1
        groups = []
        current_group = []

        for token_id, mask in zip(input_token.cpu()[0], masked_pred_labels.cpu()[0]):
            if mask == 1:
                current_group.append(token_id.item())
            elif current_group:
                groups.append(current_group)
                current_group = []

        # if masked_pred_labels ends in a 1
        if current_group:
            groups.append(current_group)

        # decode each group into a string
        keywords = [tokenizer.decode(group, 
                                     skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=True) for group in groups]

        return keywords if keywords else input