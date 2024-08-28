import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import language_tool_python
import ssl
import textstat
from transformers import BertTokenizer, BertModel
import numpy as np
import torch


from sklearn.metrics.pairwise import cosine_similarity


# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    # Get the mean pooling of the token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

def evaluate_responses(gpt_json_file_path, human_json_file_path, individual_output_file_path, cumulative_output_file_path):
    # Load the GPT responses
    with open(gpt_json_file_path, 'r') as file:
        gpt_data = json.load(file)
    gpt_responses = gpt_data.get("gpt_responses", [])

    # Load the human responses
    with open(human_json_file_path, 'r') as file:
        human_data = json.load(file)
    human_responses_dict = {entry['id']: entry['human_response'] for entry in human_data['prompts']}

    # Initialize the evaluation results
    individual_evaluation_results = []
    cumulative_scores = {}

    # Initialize the grammar checker
    tool = language_tool_python.LanguageToolPublicAPI('en-US')

    # ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Perform evaluation
    for gpt_entry in gpt_responses:
        gpt_id = gpt_entry.get("id")
        prompt = gpt_entry.get("prompt", "")
        batch_number = gpt_entry.get("batch_number", "")
        round_number = gpt_entry.get("round_number", "")
        gpt_response = gpt_entry.get("gpt_response", "")
        human_response = human_responses_dict.get(gpt_id, "")

        if not gpt_response or not human_response:
            print(f"Skipping comparison for ID {gpt_id} due to missing data.")
            continue

        # Calculate BLEU score
        reference = [nltk.word_tokenize(human_response)]
        candidate = nltk.word_tokenize(gpt_response)
        smoothing_function = SmoothingFunction().method4
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

        # Calculate ROUGE scores
        rouge_scores = rouge_scorer_obj.score(human_response, gpt_response)

        # Grammar check (count number of errors)
        grammar_errors = len(tool.check(gpt_response))

        # Readability metrics
        flesch_kincaid_grade_score = textstat.flesch_kincaid_grade(gpt_response)
        flesch_reading_ease_score = textstat.flesch_reading_ease(gpt_response)

        # Context similarity using BERT embeddings and cosine similarity
        gpt_embedding = get_sentence_embedding(gpt_response)
        human_embedding = get_sentence_embedding(human_response)
        context_similarity = cosine_similarity(gpt_embedding, human_embedding).flatten()[0]

        # Convert float32 to float for JSON serialization
        bleu_score = float(bleu_score)
        rouge_1_score = float(rouge_scores['rouge1'].fmeasure)
        rouge_2_score = float(rouge_scores['rouge2'].fmeasure)
        rouge_L_score = float(rouge_scores['rougeL'].fmeasure)
        context_similarity = float(context_similarity)
        grammar_errors = int(grammar_errors)
        flesch_kincaid_grade_score = float(flesch_kincaid_grade_score)
        flesch_reading_ease_score = float(flesch_reading_ease_score)

        # Accumulate scores for cumulative calculation
        if gpt_id not in cumulative_scores:
            cumulative_scores[gpt_id] = {
                "bleu_scores": [],
                "rouge_1_scores": [],
                "rouge_2_scores": [],
                "rouge_L_scores": [],
                "context_similarities": [],
                "total_grammar_errors": 0,
                "flesch_kincaid_grade_scores": [],
                "flesch_reading_ease_scores": [],
                "num_responses": 0
            }

        cumulative_scores[gpt_id]["bleu_scores"].append(bleu_score)
        cumulative_scores[gpt_id]["rouge_1_scores"].append(rouge_1_score)
        cumulative_scores[gpt_id]["rouge_2_scores"].append(rouge_2_score)
        cumulative_scores[gpt_id]["rouge_L_scores"].append(rouge_L_score)
        cumulative_scores[gpt_id]["context_similarities"].append(context_similarity)
        cumulative_scores[gpt_id]["total_grammar_errors"] += grammar_errors
        cumulative_scores[gpt_id]["flesch_kincaid_grade_scores"].append(flesch_kincaid_grade_score)
        cumulative_scores[gpt_id]["flesch_reading_ease_scores"].append(flesch_reading_ease_score)
        cumulative_scores[gpt_id]["num_responses"] += 1

        # Save individual evaluation results
        individual_evaluation_results.append({
            "id": gpt_id,
            "prompt": prompt,
            "batch_number": batch_number,
            "round_number": round_number,
            "bleu_score": bleu_score,
            "rouge_1": rouge_1_score,
            "rouge_2": rouge_2_score,
            "rouge_L": rouge_L_score,
            "context_similarity": context_similarity,
            "grammar_errors": grammar_errors,
            "flesch_kincaid_grade": flesch_kincaid_grade_score,
            "flesch_reading_ease": flesch_reading_ease_score,
            "gpt_response": gpt_response,
            "human_response": human_response
        })

    # Calculate cumulative scores
    cumulative_evaluation_results = []
    for gpt_id, scores in cumulative_scores.items():
        num_responses = scores["num_responses"]
        cumulative_bleu_score = sum(scores["bleu_scores"]) / num_responses
        cumulative_rouge_1_score = sum(scores["rouge_1_scores"]) / num_responses
        cumulative_rouge_2_score = sum(scores["rouge_2_scores"]) / num_responses
        cumulative_rouge_L_score = sum(scores["rouge_L_scores"]) / num_responses
        average_context_similarity = sum(scores["context_similarities"]) / num_responses
        average_grammar_errors = scores["total_grammar_errors"] / num_responses
        average_flesch_kincaid_grade = sum(scores["flesch_kincaid_grade_scores"]) / num_responses
        average_flesch_reading_ease = sum(scores["flesch_reading_ease_scores"]) / num_responses

        # Convert cumulative scores to float for JSON serialization
        cumulative_bleu_score = float(cumulative_bleu_score)
        cumulative_rouge_1_score = float(cumulative_rouge_1_score)
        cumulative_rouge_2_score = float(cumulative_rouge_2_score)
        cumulative_rouge_L_score = float(cumulative_rouge_L_score)
        average_context_similarity = float(average_context_similarity)
        average_grammar_errors = float(average_grammar_errors)
        average_flesch_kincaid_grade = float(average_flesch_kincaid_grade)
        average_flesch_reading_ease = float(average_flesch_reading_ease)

        cumulative_evaluation_results.append({
            "id": gpt_id,
            "cumulative_bleu_score": cumulative_bleu_score,
            "cumulative_rouge_1": cumulative_rouge_1_score,
            "cumulative_rouge_2": cumulative_rouge_2_score,
            "cumulative_rouge_L": cumulative_rouge_L_score,
            "average_context_similarity": average_context_similarity,
            "average_grammar_errors": average_grammar_errors,
            "average_flesch_kincaid_grade": average_flesch_kincaid_grade,
            "average_flesch_reading_ease": average_flesch_reading_ease,
            "human_response": human_responses_dict[gpt_id]
        })

    # Save individual evaluation results to a JSON file
    with open(individual_output_file_path, 'w') as outfile:
        json.dump(individual_evaluation_results, outfile, indent=4)

    # Save cumulative evaluation results to a separate JSON file
    with open(cumulative_output_file_path, 'w') as outfile:
        json.dump(cumulative_evaluation_results, outfile, indent=4)

    print(f"Individual evaluation results saved to {individual_output_file_path}")
    print(f"Cumulative evaluation results saved to {cumulative_output_file_path}")

def main():
    # Paths to GPT and Human response JSON files
    gpt_responses_file = '/Users/venus/PycharmProjects/enzymes/responses/response_problem/gpt_problem/gpt_responses_problem_and_solution_batch_1.json'
    human_responses_file = '/Users/venus/PycharmProjects/enzymes/responses/response_problem/reference_problem/reference_problem.json'

    # Output files for evaluation results
    individual_output_file = '/Users/venus/PycharmProjects/enzymes/evaluation_results/results_problem_solution/problem_individual_results.json'
    cumulative_output_file = '/Users/venus/PycharmProjects/enzymes/evaluation_results/results_problem_solution/problem_cumulative_results.json'

    # Perform evaluation
    evaluate_responses(gpt_responses_file, human_responses_file, individual_output_file, cumulative_output_file)

if __name__ == "__main__":
    main()
