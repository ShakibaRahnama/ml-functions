# Import necessary libraries
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, jaccard_score, confusion_matrix
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from rouge_score import rouge_scorer
import os
import argparse
import time


# Preprocess Notes
def preprocess_notes(df, text_column, new_column='Cleaned_Clinical_Note'):
    """
    Clean and preprocess text in a specified column of a DataFrame.
    - Converts text to lowercase.
    - Removes non-alphanumeric characters except spaces.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str): The name of the column to preprocess.
        new_column (str): The name of the new column to store cleaned text.

    Returns:
        pd.DataFrame: The DataFrame with the new column containing cleaned text.
    """
    
    # Rename columns
    df.rename(columns={
        "No.": "No.",
        "Note Type": "Note_type",
        "Full NOTE_TEXT": "Clinical_Note",
        "Review Result(Which Language speaking - Unknown=have language barrier, but which specific one is unknown)": "Review_result",
        "Classification Ground Truth": "Classification_ground_truth",
        "Relevant Span - 2Revised0207": "Relevant_span"
    }, inplace=True)
    
    df[new_column] = df[text_column].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    # Ensure 'Relevant Span' is set to 'None' where Classification = 0
    df.loc[df['Classification_ground_truth'] == 0, 'Relevant_span'] = 'None'
    
    # # Print clinical notes for first 5 patients
    # print("Clinical Notes for the First 5 Patients:")
    # for i, row in df.head(10).iterrows():
    #     print(f"\nPatient {i + 1}:")
    #     print(f"Original Note: {row['Clinical_Note']}")
    #     print("")
    #     print(f"Cleaned Note: {row['Cleaned_Clinical_Note']}")
    #     print("")
    #     print("")
    
    return df


# Create a custom dataset for clinical notes
class ClinicalNoteDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "Cleaned_Clinical_Note": str(row["Cleaned_Clinical_Note"]),
            "Classification_ground_truth": int(row["Classification_ground_truth"]),
            "Relevant_span": str(row["Relevant_span"])
        }
    

# Split Dataset
def split_data(df, label_column, seed=42):
    """
    Split data into train (15%), validation (70%), and test (15%) sets.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.8, stratify=df[label_column], random_state=seed   # 0.8
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.25, stratify=temp_df[label_column], random_state=seed   # 0.25
    )
    return train_df, val_df, test_df


def extract_answer_section(output_text):
    """
    Extracts classification, relevant span, language, and reasoning from the LLM response.
    """
    # Extract classification
    classification_match = re.search(r"Classification:\s*(Yes|No)", output_text, re.IGNORECASE)
    classification = classification_match.group(1).strip() if classification_match else "Unknown"

    # Extract relevant span (allow multiline and ensure flexible matching)
    span_match = re.search(r"Span:\s*\"?(.+?)\"?(?=\s*Language:|$)", output_text, re.IGNORECASE | re.DOTALL)
    relevant_span = span_match.group(1).strip() if span_match and span_match.group(1).strip() else "None"

    # Extract language (more robust multiline handling)
    language_match = re.search(r"Language:\s*\"?(.+?)\"?(?=\s*Reasoning:|$)", output_text, re.IGNORECASE | re.DOTALL)
    language = language_match.group(1).strip() if language_match and language_match.group(1).strip() else "None"

    # Extract reasoning (handle multiline responses)
    reasoning_match = re.search(r"Reasoning:\s*(.+)", output_text, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match and reasoning_match.group(1).strip() else "None"

    return classification, relevant_span, language, reasoning


# Define custom stopping criteria
class CustomStop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Stop when all expected sections are detected
        if re.search(
            r"Classification:\s*(Yes|No)\s*"
            r"Span:\s*\"?.+?\"?\s*"
            r"Language:\s*\"?.*?\"?\s*"
            r"Reasoning:\s*.+?\.",  # Ensure reasoning ends with a period
            output_text,
            re.IGNORECASE | re.DOTALL
        ):
            return True
        return False


# Batch Classification + Span Extraction
def classify_and_extract_with_llm(llama_pipeline, queries, stopping_criteria, train_df, max_new_tokens=10, k=3, seed=42, batch_size=4):
    """
    Use LLM for batch classification and span extraction with few-shot examples.
    """

    # Sample `k` positive and negative examples
    positive_examples = train_df[train_df["Classification_ground_truth"] == 1].sample(min(k, len(train_df)), random_state=seed)
    negative_examples = train_df[train_df["Classification_ground_truth"] == 0].sample(min(k, len(train_df)), random_state=seed)
    sampled_examples = pd.concat([positive_examples, negative_examples])

    # Construct few-shot examples
    few_shot_examples = ""
    for idx, row in enumerate(sampled_examples.itertuples(index=False), start=1):
        few_shot_examples += f"""
        ### Example {idx}: {"language barrier present" if row.Classification_ground_truth == 1 else "language barrier absent"} ###
        Clinical Note: "{row.Cleaned_Clinical_Note}"
        Classification: {"Yes" if row.Classification_ground_truth == 1 else "No"}
        Span: "{row.Relevant_span}"
        """

    # Create batch prompts
    prompts = [
        f"""
        ### Instruction ###
        Given the following clinical note, your job is to determine if the patient has a language barrier. 
        Perform the following tasks:
        **Classification:** Classify whether the note contains a language barrier marker (Answer: "Yes" or "No"). 
        **Span:** Extract the relevant span (ONLY if classification is "Yes"). "Span" here refers to the specific text segment that indicates the presence of a language barrier.
        **Language:** If the patient speaks a non-English language, mention it. 
        **Reasoning:** Provide a one-line reasoning for the classification and span extraction.

        {few_shot_examples}
        
        ### Clinical Note ###
        {query}

        ### Answer ### 
        """
        for query in queries
    ]

    # Batch generation
    try:
        t1 = time.time()
        responses = llama_pipeline(prompts, 
                                   max_new_tokens=max_new_tokens,
                                   return_full_text=False,
                                   do_sample=False,
                                   temperature=None,
                                   top_p=None,
                                #    early_stopping=True,
                                   batch_size=batch_size,
                                   pad_token_id=llama_pipeline.tokenizer.pad_token_id,
                                   eos_token_id=llama_pipeline.tokenizer.eos_token_id,
                                   stopping_criteria=stopping_criteria,
                                #    num_beams=3,
                                #    length_penalty=0.8
                                   )
        print(f"Runtime for batch inference: {time.time() - t1:.2f} seconds")
    except Exception as e:
        print(f"Error during batch inference: {e}")
        return []

    # Extract results
    results = []
    for query, response in zip(queries, responses):
        if response and isinstance(response, list) and "generated_text" in response[0]:
            output_text = response[0]["generated_text"].strip()
            classification, span, language, reasoning = extract_answer_section(output_text)
            results.append({
                "query": query,
                "classification": classification,
                "relevant_span": span,
                "language": language,
                "reasoning": reasoning,
                "llama_response": output_text
            })
        else:
            # Handle invalid responses
            print(f"Warning: No valid response generated for query: {query}")
            results.append(None)

    return results


# Compute Extraction Precision (Exact Match, ROUGE-L, Jaccard) 
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calculate_extraction_metrics(extracted_spans, ground_truth_spans, jaccard_threshold=0.5, rouge_threshold=0.7):
    """
    Compute Exact Match Precision (EMP), Weighted Exact Match Precision (WEMP), Jaccard Precision and ROUGE-L Precision 
    for span extraction.

    WEMP gives partial credit to predictions based on how much of the ground truth they match.

    Args:
        extracted_spans (list of str): Predicted spans.
        ground_truth_spans (list of str): Ground truth spans.
        rouge_threshold (float): Minimum ROUGE-L f-measure score to consider a match.
        jaccard_threshold (float): Minimum Jaccard score to consider a match.

    Returns:
        dict: 
            - "Exact Match Precision" (float)
            - "Weighted Exact Match Precision" (float)
            - "Jaccard Precision" (float)
            - "ROUGE-L Precision" (float)
    """
    exact_correct = 0
    wemp_no_penalty_correct = 0  # Weighted Exact Match Precision (WEMP-NoPenalty)
    jaccard_correct = 0
    rouge_correct = 0
    total_extractions = sum([1 if pred else 0 for pred in extracted_spans])

    for pred, gt in zip(extracted_spans, ground_truth_spans):
        if gt and pred:
            pred_tokens = pred.split()
            gt_tokens = gt.split()

            # Exact Match
            if pred == gt:
                exact_correct += 1
                
            # WEMP-NoPenalty: Doesn't penalize extra words
            pred_tokens = set(pred.split())
            gt_tokens = set(gt.split())
            wemp_no_penalty = len(pred_tokens & gt_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0
            wemp_no_penalty_correct += wemp_no_penalty

            # Jaccard Similarity Score: Ignores order of words, better for partial matches
            jaccard = len(set(pred_tokens) & set(gt_tokens)) / len(set(pred_tokens) | set(gt_tokens)) if len(set(pred_tokens) | set(gt_tokens)) > 0 else 0
            if jaccard >= jaccard_threshold:
                jaccard_correct += 1
            
            # ROUGE-L Score: Considers order of words, captures 'full' matches
            rouge_score = scorer.score(gt, pred)['rougeL'].fmeasure
            if rouge_score >= rouge_threshold:
                rouge_correct += 1

    precision_exact = exact_correct / total_extractions if total_extractions > 0 else 0
    precision_wemp_no_penalty = wemp_no_penalty_correct / total_extractions if total_extractions > 0 else 0
    precision_jaccard = jaccard_correct / total_extractions if total_extractions > 0 else 0
    precision_rouge = rouge_correct / total_extractions if total_extractions > 0 else 0

    return {
        "Exact Match Precision": precision_exact,
        "WEMP-NoPenalty": precision_wemp_no_penalty,
        "Jaccard Precision": precision_jaccard,
        "ROUGE-L Precision": precision_rouge
    }


# Evaluation with batch inference
def evaluate_performance(val_loader, train_df, llama_pipeline, stopping_criteria, max_new_tokens=10, k=3, batch_size=4, 
                         seed=42, path_viz=None, viz=False, run=None):
    all_results = []
    
    # Get global rank and world size
    global_rank = dist.get_rank()  # Unique rank across all GPUs
    world_size = dist.get_world_size()  # Total number of GPUs
    
    # Iterate over batches
    for batch_idx, batch in enumerate(val_loader):
        # Ensure batch is a list of dictionaries
        if not isinstance(batch, list):
            print(f"Unexpected batch format: {type(batch)}")
            continue

        # Extract fields
        queries = [item["Cleaned_Clinical_Note"] for item in batch]
        ground_truth_classes = ["Yes" if item["Classification_ground_truth"] == 1 else "No" for item in batch]
        ground_truth_spans = [item["Relevant_span"] for item in batch]

        # LLM Inference
        results = classify_and_extract_with_llm(
            llama_pipeline, queries, stopping_criteria, train_df,
            max_new_tokens=max_new_tokens, k=k, seed=seed, batch_size=batch_size
        )

        # Log and collect results
        for local_patient_idx, (result, query, true_class, true_span) in enumerate(zip(results, queries, ground_truth_classes, ground_truth_spans)):
            # Calculate unique patient ID
            global_patient_id = global_rank * len(val_loader.dataset) // world_size + (batch_idx * batch_size) + local_patient_idx + 1

            # Extract LLM response components
            pred_class = result['classification']
            pred_span = result['relevant_span']
            language = result['language']
            reasoning = result['reasoning']

            # Convert classification to language barrier output
            clf_pred = "SDoH" if pred_class.lower() == "yes" else "No SDoH"
            clf_gt = "SDoH" if true_class.lower() == "yes" else "No SDoH"

            # Print patient-level details
            print(f"\n🔹 Patient #{global_patient_id}")
            print(f"📝 Clinical Note: {query}")
            # print(f"🤖 LLM Response: {result['llama_response']}")
            print(f"✅ Classification prediction: {clf_pred}")
            print(f"🎯 Classification ground truth: {clf_gt}")
            print(f"🟢 Span Extracted: {pred_span}")
            print(f"🎓 Span Ground Truth: {true_span}")
            print(f"🌍 Language: {language}")
            print(f"💡 Reasoning: {reasoning}\n")

            all_results.append((result, true_class, true_span))

    # If no results, return zero metrics
    if not all_results:
        print("No valid results. Returning zero metrics.")
        return 0, {}

    # Extract predictions
    y_pred = [1 if res["classification"].lower() == "yes" else 0 for res, _, _ in all_results]
    ground_truth_classes = [1 if gt == "Yes" else 0 for _, gt, _ in all_results]
    extracted_spans = [res["relevant_span"] for res, _, _ in all_results]
    ground_truth_spans = [span for _, _, span in all_results]

    # Calculate F1 and extraction metrics
    f1 = f1_score(ground_truth_classes, y_pred)
    extraction_metrics = calculate_extraction_metrics(extracted_spans, ground_truth_spans)

    return f1, extraction_metrics


# Custom collate function to prevent tensor conversion
def collate_fn(batch):
    return batch


# Main Workflow
def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run LLaMA-3.1-8B-Instruct experiments for SDoH project.")
    parser.add_argument("--n_examps", type=int, nargs="+", default=[1, 3, 5, 10], 
                        help="List of k values for few-shot examples.")
    parser.add_argument("--max_tokens", type=int, nargs="+", default=[10, 15, 20, 25, 30], 
                        help="List of max_new_tokens values.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--n_runs", type=int, default=10, 
                        help="Number of runs with changing random seeds.")
    
    args = parser.parse_args()

    # User-defined variables
    path_g = f"/cluster/projects/brudnogroup/ani/SDoH"
    file_path = f"{path_g}/Raw_data/Language Barrier(200) - withFullText0220.xlsx"
    llm_model = "Llama-3.1-8B-Instruct"
    llm_model_path = f"{path_g}/Models/meta-llama/{llm_model}"   
    batch_size = args.batch_size

    # Load dataset
    df = pd.read_excel(file_path)
    df = preprocess_notes(df, text_column='Clinical_Note', new_column='Cleaned_Clinical_Note')
    
    # Initialize Distributed Processing
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}  # Assign each process to its GPU
    )
        
    # Inference pipeline
    stopping_criteria = StoppingCriteriaList([CustomStop(tokenizer)])  # Pass tokenizer here
    llama_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        stopping_criteria=stopping_criteria  # Add stopping criteria here
    )

    # Split data
    train_df, val_df, test_df = split_data(df, label_column='Classification_ground_truth', seed=42)
    val_df = val_df.sample(n=8, random_state=42)
    
    # Create Dataset and Distributed Sampler
    val_dataset = ClinicalNoteDataset(val_df)
    sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=False)
    
    # DataLoader for distributed processing
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        collate_fn=collate_fn  # Ensures dictionaries are retained
    )
    
    # Save train data only from rank 0 to avoid conflicts
    if local_rank == 0:
        train_df.to_excel(f"{path_g}/Data/train_data.xlsx", index=False)
    
    # Run experiments for each k and max_new_tokens combination
    for k_fs in args.n_examps:
        for max_new_tokens in args.max_tokens:
            if local_rank == 0:
                print(f"\n===== Running Experiments with k={k_fs}, max_new_tokens={max_new_tokens} =====")
            path_viz = f"{path_g}/Results/Llama-3.1-8B-Instruct/FSP/n_{k_fs}_tokens_{max_new_tokens}"
            os.makedirs(path_viz, exist_ok=True)
            
            # Store performance results
            f1_scores, exact_precisions, weighted_exact_precisions, jaccard_scores, rouge_scores = [], [], [], [], []
            
            for run in range(args.n_runs):
                if local_rank == 0:
                    print(f"\n----- Run {run + 1}/{args.n_runs} for k={k_fs}, max_new_tokens={max_new_tokens} -----")
                seed = 42 + run
                
                # Evaluate on validation set
                f1, extraction_metrics = evaluate_performance(
                    val_loader, train_df, llama_pipeline, stopping_criteria,
                    max_new_tokens=max_new_tokens, k=k_fs, batch_size=batch_size, 
                    seed=seed, path_viz=path_viz, viz=False, run=run
                )
                
                # Store results (only rank 0 logs results)
                if local_rank == 0:
                    f1_scores.append(f1)
                    exact_precisions.append(extraction_metrics["Exact Match Precision"])
                    weighted_exact_precisions.append(extraction_metrics["WEMP-NoPenalty"])
                    jaccard_scores.append(extraction_metrics["Jaccard Precision"])
                    rouge_scores.append(extraction_metrics["ROUGE-L Precision"])

            # # Print final results for this combination
            # if local_rank == 0:
            #     print("\n===== Final Performance for k={} and max_new_tokens={} =====".format(k_fs, max_new_tokens))
            #     for metric, values in zip(["F1-score", "Exact Precision", "WEMP-NoPenalty", "Jaccard Precision", "ROUGE-L Precision"], 
            #                             [f1_scores, exact_precisions, weighted_exact_precisions, jaccard_scores, rouge_scores]):
            #         print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
                    
            #     # Define output file for results
            #     result_file = f"{path_viz}/final_results_k{k_fs}_tokens{max_new_tokens}.txt"

            #     # Write results to text file
            #     with open(result_file, "w") as f:
            #         f.write(f"===== Final Performance for k={k_fs} and max_new_tokens={max_new_tokens} =====\n")
            #         for metric, values in zip(
            #             ["F1-score", "Exact Precision", "WEMP-NoPenalty", "Jaccard Precision", "ROUGE-L Precision"], 
            #             [f1_scores, exact_precisions, weighted_exact_precisions, jaccard_scores, rouge_scores]
            #         ):
            #             mean_score = np.mean(values)
            #             std_score = np.std(values)
            #             f.write(f"{metric}: {mean_score:.4f} ± {std_score:.4f}\n")
                        
    dist.barrier()
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
