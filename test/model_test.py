import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./keT5/model_epoch_1/"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-base-ko")
# tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

model.eval()
model.to(device)

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(input_ids=input_ids, max_length=100)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

try:
    while True:
        user_input = input("User: ")
        model_response = generate_response(user_input)
        print("AI: ", model_response)
except KeyboardInterrupt:
    print("Ending the conversation.")
