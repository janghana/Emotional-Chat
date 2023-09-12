from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-base-ko")
model = T5ForConditionalGeneration.from_pretrained('./keT5/Emotional_Chat_keT5.pth')

def test_model(input_sentence):
    input_sentence = "대화: " + input_sentence  # Ensure that the format is correct
    inputs = tokenizer.encode_plus(
        input_sentence, 
        max_length=512, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=100,
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
        )

    predicted_sentence = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return predicted_sentence

input_sentence = "안녕하세요, 오늘 날씨는 어때요?"
print(test_model(input_sentence))
