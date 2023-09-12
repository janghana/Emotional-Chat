from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

def evaluate(model, tokenizer, input_texts, output_texts):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing_function = SmoothingFunction().method1

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    bleu_scores = []
    num_samples = len(input_texts)

    with torch.no_grad():
        for input_text, output_text in tqdm(zip(input_texts, output_texts), total=num_samples, desc='Evaluating'):
            inputs = tokenizer.encode_plus(
                input_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            targets = tokenizer.encode_plus(
                output_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            predicted_tokens = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
            predicted_sentences = [tokenizer.decode(tokens) for tokens in predicted_tokens]
            ground_truth_sentences = [tokenizer.decode(tokens) for tokens in targets.input_ids]

            rouge_scores = [scorer.score(pred, gt) for pred, gt in zip(predicted_sentences, ground_truth_sentences)]
            bleu_score = sentence_bleu([ground_truth_sentences], predicted_sentences[0], smoothing_function=smoothing_function)

            for metric, score in rouge_scores[0].items():
                scores[metric] += score.fmeasure

            bleu_scores.append(bleu_score)

    for metric in scores:
        scores[metric] /= num_samples

    avg_bleu_score = sum(bleu_scores) / num_samples

    return scores, avg_bleu_score

if __name__ == '__main__':
    model_path = './keBART/Emotional_Chat-epoch=01-val_loss=0.05.tmp_end.ckpt'
    test_data_path = './data/Test.json'

    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
    model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    input_texts, output_texts = load_data(test_data_path)

    scores, avg_bleu_score = evaluate(model, tokenizer, input_texts, output_texts)

    print('ROUGE scores:')
    for metric, score in scores.items():
        print(f'{metric}: {score:.4f}')

    print(f'Average BLEU score: {avg_bleu_score:.4f}')
