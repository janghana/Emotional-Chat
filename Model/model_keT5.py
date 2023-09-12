from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AdamW, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
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
    def __init__(self, tokenizer):
        super(ConversationModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-base-ko")
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.tokenizer = tokenizer  # tokenizer is added here

    def forward(self, input_ids, attention_mask, labels=None):
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

        # Use the model to generate predictions
        outputs = self(input_ids, attention_mask, labels)
        predicted_ids = self.model.generate(input_ids, attention_mask=attention_mask)
        predicted_text = [self.tokenizer.decode(ids) for ids in predicted_ids]
        label_text = [self.tokenizer.decode(ids) for ids in labels]
        return {'predicted_text': predicted_text, 'label_text': label_text}

    def validation_epoch_end(self, outputs):
        rouge_score = 0
        bleu_score = 0
        n = 0
        for output in outputs:
            predicted_texts = output['predicted_text']
            label_texts = output['label_text']
            for predicted_text, label_text in zip(predicted_texts, label_texts):
                scores = self.scorer.score(label_text, predicted_text)
                rouge_score += sum([scores[key].fmeasure for key in scores.keys()])
                bleu_score += sentence_bleu([label_text.split()], predicted_text.split())
                n += 1

        rouge_score = rouge_score / n
        bleu_score = bleu_score / n

        self.log('rouge_score', rouge_score, on_epoch=True, prog_bar=True, logger=True)
        self.log('bleu_score', bleu_score, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)

    def on_fit_end(self):
        torch.save(self.state_dict(), './keT5/Emotional_Chat_keT5.pth')


if __name__ == '__main__':
    freeze_support()

    input_texts, output_texts = load_data('./data/Training.json')

    tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base-ko")

    split_index = int(len(input_texts) * 0.8)
    train_input_texts, val_input_texts = input_texts[:split_index], input_texts[split_index:]
    train_output_texts, val_output_texts = output_texts[:split_index], output_texts[split_index:]

    train_dataset = ConversationDataset(tokenizer, train_input_texts, train_output_texts)
    val_dataset = ConversationDataset(tokenizer, val_input_texts, val_output_texts)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./keT5', 
        filename='Emotional_Chat-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1, 
        verbose=True, 
        monitor='val_loss', 
        mode='min'
    )

    wandb_logger = WandbLogger(name="Emotioanl_Chat", project="Emotioanl_Chat")

    model = ConversationModel(tokenizer)
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=2, 
        progress_bar_refresh_rate=20, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader)

