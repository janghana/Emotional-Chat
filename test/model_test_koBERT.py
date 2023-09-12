import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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


def test_model(model, dataloader):
    # Ensure the model is in evaluation mode
    model.eval()

    # If you have a GPU available, move your model to it
    if torch.cuda.is_available():
        model.to('cuda')

    # Choose the first batch from the dataloader for testing
    batch = next(iter(dataloader))

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    # If you have a GPU, also move your inputs to it
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        labels = labels.to('cuda')

    # Generate the model's prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss

    # Print loss for the current batch
    print(f"Loss: {loss.item()}")
    
    # Generate some text based on the input
    predicted_ids = model.model.generate(input_ids, attention_mask=attention_mask)
    predicted_text = [model.tokenizer.decode(ids) for ids in predicted_ids]

    print(f"Generated text: {predicted_text}")


if __name__ == "__main__":
    input_texts, output_texts = load_data('./data/Training.json')
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    train_dataset = ConversationDataset(tokenizer, input_texts, output_texts)
    dataloader = DataLoader(train_dataset, batch_size=2)
    model = ConversationModel(tokenizer)
    
    # Load model weights from file
    model.load_state_dict(torch.load("./koBERT/koBERT_Chat.pth"))
    
    test_model(model, dataloader)
