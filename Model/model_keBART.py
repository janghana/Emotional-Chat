
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import json
import torch
from tqdm import tqdm
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_texts = []
    output_texts = []

    for conversation in data:
        talk_content = conversation['talk']['content']
        
        human_sentences = [talk_content[key] for key in talk_content if key.startswith('HS')]
        ai_responses = [talk_content[key] for key in talk_content if key.startswith('SS')]

        for human, ai in zip(human_sentences, ai_responses):
            input_texts.append(human)
            output_texts.append(ai)
    
    return input_texts, output_texts

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, input_texts, output_texts, max_length=512):
        self.tokenizer = tokenizer
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        inputs = self.tokenizer.encode_plus(
            input_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        targets = self.tokenizer.encode_plus(
            output_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten(),
        }

input_texts, output_texts = load_data('./data/Training.json')

tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
 
# tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-small")
 
# model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-small")

dataset = ConversationDataset(tokenizer, input_texts, output_texts)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
# make sure GPU
print(torch.cuda.is_available())
print(device)
print(torch.cuda.device_count())

print(model.device_ids)

epochs = 1
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.mean().item()

        loss.mean().backward()
        optimizer.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.mean().item()/len(batch))})

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Average Loss = {average_loss:.4f}")

    model.module.save_pretrained(f'./model_epoch_{epoch+1}')
