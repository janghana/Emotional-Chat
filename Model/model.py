from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_texts = []
    output_texts = []

    for conversation in data:
        talk_content = conversation['talk']['content']
        
        # extract the human sentences and the AI responses
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-small-ko")
    model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-base-ko")
    
    # load the data
    input_texts, output_texts = load_data('./data/Training.json')

    dataset = ConversationDataset(tokenizer, input_texts, output_texts)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


    input_text = "Translate this sentence"
    output_text = "이 문장을 번역하세요"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)  # move the input tensors to the device
    output_ids = tokenizer.encode(output_text, return_tensors="pt").to(device)  # move the output tensors to the device


    loss = model(input_ids=input_ids, labels=output_ids).loss

    # Save model
    torch.save(trainer.model.state_dict(), 'model_path.pt')
