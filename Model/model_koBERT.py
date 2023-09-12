from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
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
import torch.multiprocessing as mp


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

        outputs = self.tokenizer.encode_plus(
            output_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': outputs['input_ids'].flatten()  # output ids are used as labels
        }


class ConversationModel(pl.LightningModule):
    def __init__(self, tokenizer):
        super(ConversationModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("monologg/kobert")
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

        outputs = self(input_ids, attention_mask, labels=labels)  # labels are provided
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Use the model to generate predictions
        outputs = self(input_ids, attention_mask, labels=labels)
        val_loss = outputs.loss  # compute validation loss
        # Use the model to generate predictions
        predicted_ids = self.model.generate(input_ids, attention_mask=attention_mask)
        predicted_text = [self.tokenizer.decode(ids) for ids in predicted_ids]

        # Since the output text is not provided in the batch, we'll assume for now that it should be the same as the input
        output_text = [self.tokenizer.decode(ids) for ids in input_ids]
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # log the validation loss

        return {'output_text': output_text, 'predicted_text': predicted_text, 'val_loss': val_loss}  # return the validation loss


    def validation_epoch_end(self, outputs):
        rouge_score = 0
        bleu_score = 0
        n = 0
        for output in outputs:
            output_texts = output['output_text']
            predicted_texts = output['predicted_text']
            for output_text, predicted_text in zip(output_texts, predicted_texts):
                scores = self.scorer.score(output_text, predicted_text)
                rouge_score += sum([scores[key].fmeasure for key in scores.keys()])
                bleu_score += sentence_bleu([output_text.split()], predicted_text.split())
                n += 1

        rouge_score = rouge_score / n
        bleu_score = bleu_score / n

        self.log('rouge_score', rouge_score, on_epoch=True, prog_bar=True, logger=True)
        self.log('bleu_score', bleu_score, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)

    def on_fit_end(self):
        torch.save(self.state_dict(), './koBERT/koBERT_Chat.pth')


if __name__ == '__main__':
    freeze_support()
    mp.set_start_method('spawn')

    input_texts, output_texts = load_data('./data/Training.json')

    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

    split_index = int(len(input_texts) * 0.8)
    train_input_texts, val_input_texts = input_texts[:split_index], input_texts[split_index:]
    train_output_texts, val_output_texts = output_texts[:split_index], output_texts[split_index:]

    train_dataset = ConversationDataset(tokenizer, train_input_texts, train_output_texts)
    val_dataset = ConversationDataset(tokenizer, val_input_texts, val_output_texts)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, pin_memory=True, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./koBERT', 
        filename='koBERT_Chat-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1, 
        verbose=True, 
        monitor='val_loss', 
        mode='min'
    )

    wandb_logger = WandbLogger(name="koBERT_Chat", project="koBERT_Chat")

    model = ConversationModel(tokenizer)
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=2, 
        progress_bar_refresh_rate=20, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader)
