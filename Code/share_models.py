from transformers import RobertaForMaskedLM, GPT2LMHeadModel, AutoTokenizer
from datasets import load_from_disk, Dataset

from huggingface_hub import HfApi
api = HfApi()

# --- This file contains functions in charge of uploading the pre-trained models and the cleaned dataset to Hugging Face --- #

def share_NewsGPT():
    pt_model = GPT2LMHeadModel.from_pretrained("models/gpt_sciride_news_rand_base/checkpoint-last")
    pt_model.push_to_hub("NewsGPT")
    tokenizer = AutoTokenizer.from_pretrained("models/gpt_sciride_news_rand_base/checkpoint-last")
    tokenizer.push_to_hub("NewsGPT")
    api.upload_file(
        path_or_fileobj="HF_model_and_dataset_cards/newsgpt_model-card.md",
        path_in_repo="README.md",
        repo_id="andyreas/newsgpt",
        repo_type="model",
    )

def share_GenNewsGPT():
    pt_model = GPT2LMHeadModel.from_pretrained("models/gpt_sciride_news_gen_base/checkpoint-last")
    pt_model.push_to_hub("GenNewsGPT")
    tokenizer = AutoTokenizer.from_pretrained("models/gpt_sciride_news_gen_base/checkpoint-last")
    tokenizer.push_to_hub("GenNewsGPT")
    api.upload_file(
        path_or_fileobj="HF_model_and_dataset_cards/gennewsgpt_model-card.md",
        path_in_repo="README.md",
        repo_id="andyreas/gennewsgpt",
        repo_type="model",
    )

def share_roberta_news():
    pt_model = RobertaForMaskedLM.from_pretrained("models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2450000 (epoch 3)")
    pt_model.push_to_hub("roberta-news")
    tokenizer = AutoTokenizer.from_pretrained("models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2450000 (epoch 3)")
    tokenizer.push_to_hub("roberta-news")
    api.upload_file(
        path_or_fileobj="HF_model_and_dataset_cards/roberta-news_model-card.md",
        path_in_repo="README.md",
        repo_id="andyreas/roberta-news",
        repo_type="model",
    )

def share_roberta_gen_news():
    pt_model = RobertaForMaskedLM.from_pretrained("models/roberta_sciride_news_gen_base/checkpoint-last")
    pt_model.push_to_hub("roberta-gen-news")
    tokenizer = AutoTokenizer.from_pretrained("models/roberta_sciride_news_gen_base/checkpoint-last")
    tokenizer.push_to_hub("roberta-gen-news")
    api.upload_file(
        path_or_fileobj="HF_model_and_dataset_cards/roberta-gen-news_model-card.md",
        path_in_repo="README.md",
        repo_id="andyreas/roberta-gen-news",
        repo_type="model",
    )

def share_full_dataset():
    dataset: Dataset = load_from_disk("arrow_datasets/filtered_dataset_nodups")
    dataset.push_to_hub("frontpage-news")
    api.upload_file(
        path_or_fileobj="HF_model_and_dataset_cards/frontpage-news_dataset-card.md",
        path_in_repo="README.md",
        repo_id="andyreas/frontpage-news",
        repo_type="dataset",
    )

def main():
    # share_NewsGPT()
    # share_GenNewsGPT()
    # share_roberta_news()
    # share_roberta_gen_news()
    # share_full_dataset()
    pass

if __name__ == "__main__":
    main()