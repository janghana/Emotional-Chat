import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_model(model_path, test_input):
    # Load the trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

    # Ensure the model is in evaluation mode
    model.eval()

    # If you have a GPU available, move your model to it
    if torch.cuda.is_available():
        model.to('cuda')

    # Tokenize the test input
    inputs = tokenizer.encode_plus(
        test_input, 
        return_tensors='pt', 
        max_length=512, 
        padding='max_length', 
        truncation=True
    )

    # If you have a GPU, also move your inputs to it
    if torch.cuda.is_available():
        inputs.to('cuda')

    # Generate the model's prediction
    with torch.no_grad():
        predicted_tokens = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Decode the model's prediction
    predicted_sentence = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
    
    return predicted_sentence


if __name__ == "__main__":
    model_path = "./keBART/Emotional_Chat_keBART.pth"
    test_input = "너의 이름이 뭐야?" # replace with your test input
    prediction = test_model(model_path, test_input)
    print(f"Input: {test_input}\nOutput: {prediction}")
