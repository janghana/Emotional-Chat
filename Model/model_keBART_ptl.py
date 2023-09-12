from transformers import AdamW, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import pytorch_lightning as pl
import json
import torch
from tqdm import tqdm
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from multiprocessing import Process, freeze_support
from pytorch_lightning.callbacks import ModelCheckpoint

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


class ConversationModel(pl.LightningModule):
    def __init__(self):
        super(ConversationModel, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask, labels)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        predicted_tokens = self.model.generate(input_ids, attention_mask=attention_mask)
        predicted_sentences = [self.tokenizer.decode(tokens) for tokens in predicted_tokens]
        ground_truth_sentences = [self.tokenizer.decode(tokens) for tokens in labels]
        
        rouge_scores = [self.rouge.score(pred, gt) for pred, gt in zip(predicted_sentences, ground_truth_sentences)]
        avg_rouge_scores = {key: sum([score[key].fmeasure for score in rouge_scores])/len(rouge_scores) for key in rouge_scores[0].keys()}
        
        bleu_scores = [sentence_bleu([gt], pred, smoothing_function=self.smoothing_function) for pred, gt in zip(predicted_sentences, ground_truth_sentences)]
        avg_bleu_score = sum(bleu_scores)/len(bleu_scores)
        
        self.log('avg_bleu_score', avg_bleu_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_rouge_scores', avg_rouge_scores, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)

    def on_fit_end(self):
        self.model.save_pretrained('./keBART/Emotional_Chat_keBART.pth')

if __name__ == '__main__':
    freeze_support()

    input_texts, output_texts = load_data('./data/Training.json')

    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

    split_index = int(len(input_texts) * 0.8)
    train_input_texts, val_input_texts = input_texts[:split_index], input_texts[split_index:]
    train_output_texts, val_output_texts = output_texts[:split_index], output_texts[split_index:]

    train_dataset = ConversationDataset(tokenizer, train_input_texts, train_output_texts)
    val_dataset = ConversationDataset(tokenizer, val_input_texts, val_output_texts)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./keBART', 
        filename='Emotional_Chat-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1, 
        verbose=True, 
        monitor='val_loss', 
        mode='min'
    )

    wandb_logger = WandbLogger(name="Emotioanl_Chat", project="Emotioanl_Chat")

    model = ConversationModel()
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=2, 
        progress_bar_refresh_rate=20, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader)
