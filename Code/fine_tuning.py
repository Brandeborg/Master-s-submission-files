import os
import torch
import random
from typing import Callable
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from data_preparation import load_json, save_json, generalize_NER_dataset, substitute_keywords_in_rows
from evaluation import handle_predictions_NER, handle_logits_multi_label, handle_logits_single_label
from data_preprocessing_finetuning import data_preprocessing_seq_class, data_preprocessing_NER, data_preprocessing_title_desc_match

# --- This file contains functions in charge of fine-tuning pre-trained models for various NLP tasks --- #

load_dotenv()
# --- LOAD ENV CONSTANTS FOR CONSISTENT FILE NAMES --- #
LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME = os.getenv("LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME")  
LABELED_COVID_CLASS_DATASET_DIR_NAME = os.getenv("LABELED_COVID_CLASS_DATASET_DIR_NAME")
LABELED_NER_DATASET_DIR_NAME = os.getenv("LABELED_NER_DATASET_DIR_NAME")
LABELED_NER_DATASET_NAMESPLIT_DIR_NAME = os.getenv("LABELED_NER_DATASET_NAMESPLIT_DIR_NAME")
LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME = os.getenv("LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME")

FINE_TUNED_MODEL_COVID_DIR_NAME = os.getenv("FINE_TUNED_MODEL_COVID_DIR_NAME")
FINE_TUNED_MODEL_COVID_SUB_DIR_NAME = os.getenv("FINE_TUNED_MODEL_COVID_SUB_DIR_NAME")
FINE_TUNED_MODEL_POLIT_DIR_NAME = os.getenv("FINE_TUNED_MODEL_POLIT_DIR_NAME")
FINE_TUNED_MODEL_POLIT_SUB_DIR_NAME = os.getenv("FINE_TUNED_MODEL_POLIT_SUB_DIR_NAME")
FINE_TUNED_MODEL_NER_DIR_NAME = os.getenv("FINE_TUNED_MODEL_NER_DIR_NAME")
FINE_TUNED_MODEL_TITLE_DESC_MATCH_DIR_NAME = os.getenv("FINE_TUNED_MODEL_TITLE_DESC_MATCH_DIR_NAME")

def training_loop(tokenized_datasets: DatasetDict, pre_checkpoint: str, model_type, save_dir, compute_metrics, preprocess_logits_for_metrics, label2id_map: dict, tokenizer, data_collator, training_seed, num_train_epochs=1):
    """Starts a model training loop using Hugging Face Trainer class. 

    Args:
        tokenized_datasets (DatasetDict): The tokenized dataset used for training
        pre_checkpoint (str): The checkpoint of the pre-trained model acting as the base of the fine-tuned model
        model_type (Type): The type of model. For instance "RobertaForSequenceClassification".
        save_dir (str): Where to save the model.
        compute_metrics (Callable): Function used during evaluation during training, passed to Trainer.
        preprocess_logits_for_metrics (Callable): Function called before compute_metrics, passed to Trainer.
        label2id_map (dict): Dict functioning as a map from label id to label string. Defaults to None.
        tokenizer (Tokenizer): Tokenizer passed to trainer, so it can be saved along with the model. 
        data_collator (DataCollator): DataCollator passed to Trainer to pack batches
        training_seed (int): A seed used to obtain determinstic results during training
        num_train_epochs (int, optional): How many epochs the training loop runs for. Defaults to 1.
    """
    from transformers import RobertaConfig, TrainingArguments, Trainer

    id2label_map = {v: k for k, v in label2id_map.items()}

    def model_init():
        """Creates and returns [model_type] model either based on a pretrained checkpoint, where only the head params are new, or from scratch, so all params are new.

        The function is passed as a callback to the Trainer, instead of an initialised model, so the Trainer's "seed" param is used when initialising the model.
        The "seed" param is used in the Trainer's set_seed() function, which sets a global torch seed. If a model is initialised before being passed to the Trainer,
        it will not use the seed when randomly choosing weights. If fine-tuning is run multiple times in a loop, it means the first model will be random, but all others
        will be based on a seed. Therefore, passing a function that will be called after set_seed() is better.
        """
        # load pretrained base model and attach untrained sequence classification head
        if pre_checkpoint != None:
            model = model_type.from_pretrained(
                    pre_checkpoint, id2label=id2label_map, label2id=label2id_map)
        # load untrained base model and attach untrained sequence classification head
        else:
            config = RobertaConfig(vocab_size=len(tokenizer),
                                max_position_embeddings=514, 
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id, 
                                id2label=id2label_map, 
                                label2id=label2id_map)
            model = model_type(config)

        ### DEBUGGING ###
        # Save some of the model head parameters in a json to see if the head params get properly, randomly initilizied on every training iteration
        # after setting a random seed in training_args
        # save_some_model_head_params(model)

        # Result: 
        # Not chossing a random seed resulted in only the first iteration having different head params. All other iterations had the same head params. 
        # Choosing a random seed resultet in different head params on every iteration.
        # Not sure how a training argument, which are used after init, can change how the model is initialised.

        # The base model(everything other than the head)'s params remain, expectedly, unchanged no matter what.
        ### DEBUGGING ###

        return model

    # define training parameters
    # if seed is None, default behaviour is non-determinism via random seed
    seed = random.randint(1, 1000000) if training_seed == None else training_seed 
    training_args = TrainingArguments(
        output_dir=save_dir + "/checkpoints",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs, # 3 for token classification, otherwise 1
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        seed=seed)

    # pass model, data, evaluation parameters, etc. to trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)

    # delete directory where old model was saved
    # this was an attempt at fixing a bug at one point, which
    # is probably not necessary, but I don't want to test that.
    import shutil
    try:
        shutil.rmtree(save_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    # run training
    trainer.train()
    trainer.save_model(save_dir + "/checkpoint-last")

    del trainer

### Sequence Classification
def fine_tune_sequence_classifier(labeled_datasets: DatasetDict, save_dir: str, label2id_map: dict, data_preprocessing: Callable, pre_checkpoint, multi_label: bool, training_seed=None):
    """Prepares the arguments needed for fine tuning a sequence classifier and passes them to training_loop().

    Args:
        labeled_datasets (DatasetDict): DatasetDict of raw text sequences and labels.
        save_dir (str): Where to save the fine-tuned model
        label2id_map (dict): Dict functioning as a map from label id to label string. Defaults to None.
        data_preprocessing (Callable): Function use to pre-process data
        pre_checkpoint (str): Checkpoint used to load pretrained model and its tokenizer
        multi_label (bool): Whether the data is multi-labeled
        training_seed (int, optional): Seed used to keep model parameters consistent. Defaults to None.
    """
    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, DataCollatorWithPadding
    import evaluate
    # clear cache before using the GPU to avoid memory crash
    torch.cuda.empty_cache()

    # load the metrics used to evaluate seq classification
    metrics = evaluate.combine([evaluate.load("accuracy"), evaluate.load("f1")])

    handle_logits = handle_logits_multi_label if multi_label else handle_logits_single_label

    # in charge of tokenizing raw text data, so it is split up into tokens and converted to ids (ints)
    tokenizer_checkpoint = pre_checkpoint if pre_checkpoint != None else "roberta-base"
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(tokenizer_checkpoint)

    # in charge of creating input batches for the model. Makes sure batch samples are equal length (padded)
    data_collator = DataCollatorWithPadding(tokenizer)

    tokenized_datasets = DatasetDict()
    tokenized_datasets["train"] = data_preprocessing(labeled_datasets["train"], tokenizer, multi_label=multi_label)
    tokenized_datasets["validation"] = data_preprocessing(labeled_datasets["validation"], tokenizer, multi_label=multi_label)
    
    tokenized_datasets = tokenized_datasets.shuffle(training_seed)

    def compute_metrics(pred):
        """Computes metrics during mid-training evaluation.

        Returns:
            dict: Evaluation results
        """
        predictions, labels = pred.predictions
        return metrics.compute(predictions = predictions, references=labels)
    
    def preprocess_logits_for_metrics(logits, labels):
        """Pre-processing to decrease memory usage of GPU. More details in fine_tune_NER() and language_modeling.py.

        Args:
            logits (list): logits produced by model
            labels (list): true labels

        Returns:
            tuple: predictions and labels
        """
        predictions, labels = handle_logits(logits, labels)
        return predictions, labels

    # run training
    training_loop(compute_metrics=compute_metrics,
                  preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
                  model_type=RobertaForSequenceClassification, 
                  tokenized_datasets=tokenized_datasets, 
                  pre_checkpoint=pre_checkpoint, 
                  save_dir=save_dir, 
                  label2id_map=label2id_map, 
                  tokenizer=tokenizer, 
                  data_collator=data_collator, 
                  training_seed=training_seed)

### Token Classification (NER)
# inspired by HF token classification tutorial
# https://huggingface.co/docs/transformers/tasks/token_classification
def fine_tune_NER(raw_dataset: Dataset, save_dir: str, label2id_map, pre_checkpoint, training_seed=None):
    """Runs preprocessing and then fine tunes a pretrained model for token classification, evaluating after each epoch.

    Args:
        raw_dataset (Dataset): A dataset of text to be tokenized before training
        label2id_map (dict): A map from label text to label id
        pre_checkpoint (str, optional): The checkpoint of the pretrained model. Defaults to "roberta-base".
    """
    from transformers import RobertaForTokenClassification
    import evaluate
    # clear cache before using the GPU to avoid memory crash
    torch.cuda.empty_cache()

    seqeval = evaluate.load("seqeval")

    id2label_map = {v: k for k, v in label2id_map.items()}
    
    # run preprocessing, tokenizing the input text into input_ids
    tokenizer_checkpoint = pre_checkpoint if pre_checkpoint != None else "roberta-base"
    tokenizer, tokenized_datasets, data_collator = data_preprocessing_NER(raw_dataset=raw_dataset, checkpoint=tokenizer_checkpoint)

    # shuffling, so when running fine-tuning on the same data multiple times in a row (for testing), the results are different
    tokenized_datasets = tokenized_datasets.shuffle()

    def compute_metrics(pred):
        """Computes metrics during mid-training evaluation

        Returns:
            dict: Evaluation results
        """
        # pred contains a "predictions" attribute, which contains the preprocessed predictions. pred also contains a "label_ids" attribute which contains the original labels
        # from before preprocessing. The labels were not touched during preprocessing, so label_ids can be used.
        predictions, labels = pred
    
        # make the predictions and labels ready for seqeval
        true_predictions, true_labels = handle_predictions_NER(predictions, labels, id2label_map)

        # prec, rec and f1 may be omitted in cases where prec and rec end up being 0, because of division by zero being undefined. It does not affect training.
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        For pre_training the preprocess function is used to reduce the memory usage. But in this case, it is used to force the input of compute_metrics
        to be consistent between fine-tuning and evaluation. For some reason, compute_metrics receives a numpy array during the mid-training evaluation, 
        but during the post-training evaluation, it receives a torch tensor. This means torch.argmax() works for one, but not the other, and vice versa for
        np.argmax()

        This function always receives tensors, so the logits are handled here, before they are passed to compute_metrics. And even though 
        the function docs specifiy that preprocess_logits_for_metrics produces a Tensor, compute_metrics still receives a np.array,
        so something happens between this function and compute_metrics.
        """
        # get label index (label id) of the higest scoring prediction of each token
        predictions = torch.argmax(logits, dim=-1)
        
        # Whatever is returned here will be in the "predictions" attribute of the eval_pred parameter in compute_metrics(eval_pred)
        return predictions

    # NER dataset is small, so the model is trained for multiple epochs
    training_loop(compute_metrics=compute_metrics,
                  preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
                  model_type=RobertaForTokenClassification, 
                  tokenized_datasets=tokenized_datasets, 
                  pre_checkpoint=pre_checkpoint, 
                  save_dir=save_dir, 
                  label2id_map=label2id_map, 
                  tokenizer=tokenizer, 
                  data_collator=data_collator, 
                  training_seed=training_seed,
                  num_train_epochs=3)

### Utility used for debugging, to see if the training_seed actually affected the model parameters.
def extract_some_model_head_parameters(model):
    params = {}
    for name, param in model.named_parameters():
        if "classifier" in name:
            params[name] = param.data.numpy().tolist()[:10]

    return params

def save_some_model_head_params(model):
    try:
        model_params = load_json("blobs/model_params.json")
    except:
        model_params = {"iterations": []}

    model_params["iterations"].append(extract_some_model_head_parameters(model))

    save_json(model_params, "blobs/model_params.json")

# wrapper functions
### --- NER --- ###
def load_data_and_fine_tune_NER(pre_checkpoint, training_seed, gen=True, name_tag=""):
    """Loads dataset and starts training NER model.

    Args:
        pre_checkpoint (str): The base model used for fine-tuning
        training_seed (int): Seed to keep model head parameters consistent
        gen (bool, optional): True if data should be generalized before training (run generalize_NER_dataset()). Defaults to True.
        name_tag (str, optional): Tag to put at the end of name where model is saved. Defaults to "".
    """
    ner_dataset = load_from_disk(LABELED_NER_DATASET_NAMESPLIT_DIR_NAME)

    if (gen):
        ner_dataset = generalize_NER_dataset(ner_dataset)
        label2id_map = {"O": 0, "B-NE": 1, "I-NE": 2}
    else:
        label2id_map = load_json("blobs/NER_labelmap.json")

    fine_tune_NER(ner_dataset, save_dir=FINE_TUNED_MODEL_NER_DIR_NAME+name_tag, label2id_map=label2id_map, pre_checkpoint=pre_checkpoint, training_seed=training_seed)

### --- COVID --- ###
def load_data_and_fine_tune_covid(n_samples, pre_checkpoint, training_seed, sub = False, name_tag=""):
    """Loads dataset and starts training covid classifier model.

    Args:
        n_samples (int): Amount of samples used for training
        pre_checkpoint (str): The base model used for fine-tuning
        training_seed (int): Seed to keep model head parameters consistent
        sub (bool, optional): True if COVID keywords should be substituted with <mask>. Defaults to False.
        name_tag (str, optional): Tag to put at the end of name where model is saved. Defaults to "".
    """
    covid_dataset = load_from_disk(LABELED_COVID_CLASS_DATASET_DIR_NAME)

    if (sub):
        covid_class_keywords = load_json("blobs/covid_class_keywords.json")
        keywords = [keyword for cls in covid_class_keywords.values() for keyword in cls["keywords"]]
        covid_dataset["train"] = substitute_keywords_in_rows(covid_dataset["train"], keywords, "<mask>")

    # select n_samples with evenly distributed labels
    covid_dataset["train"] = covid_dataset["train"].select(range(n_samples*2))
    positive_and_negative = [covid_dataset["train"].filter(lambda example: example["label"] == label).select(range(int(n_samples/2))) for label in [1,0]]
    covid_dataset["train"] = concatenate_datasets(positive_and_negative).shuffle(seed=2022)

    class_keywords = load_json("blobs/covid_class_keywords.json")
    label2id_map = {key: i for i, key in enumerate(class_keywords.keys())}
    fine_tune_sequence_classifier(covid_dataset, save_dir=FINE_TUNED_MODEL_COVID_DIR_NAME+name_tag, label2id_map=label2id_map, 
                                data_preprocessing = data_preprocessing_seq_class, pre_checkpoint=pre_checkpoint, multi_label=False, training_seed=training_seed)

### --- POLITICIANS --- ###
def load_data_and_fine_tune_multi(n_samples, pre_checkpoint, training_seed, sub=False, name_tag=""):
    """Loads dataset and starts training politician classifier model.

    Args:
        n_samples (int): Not used because a custom size dataset is loaded
        pre_checkpoint (str): The base model used for fine-tuning
        training_seed (int): Seed to keep model head parameters consistent
        sub (bool, optional): True if COVID keywords should be substituted with <mask>. Defaults to False.
        name_tag (str, optional): Tag to put at the end of name where model is saved. Defaults to "".
    """
    multi_dataset = load_from_disk(LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME)

    if (sub):
        multi_class_keywords = load_json("blobs/multi_class_keywords.json")
        keywords = [keyword for cls in multi_class_keywords.values() for keyword in cls["keywords"]]
        multi_dataset["train"] = substitute_keywords_in_rows(multi_dataset["train"], keywords, "<mask>")

    multi_dataset["train"] = multi_dataset["train"].shuffle(seed=2022)
    class_keywords = load_json("blobs/multi_class_keywords.json")
    label2id_map = {key: i for i, key in enumerate(class_keywords.keys())}
    fine_tune_sequence_classifier(multi_dataset, save_dir=FINE_TUNED_MODEL_POLIT_DIR_NAME+name_tag, label2id_map=label2id_map, 
                                data_preprocessing = data_preprocessing_seq_class, pre_checkpoint=pre_checkpoint, multi_label=True, training_seed=training_seed)

### --- TITLE-DESC MATCH --- ###
def load_data_and_fine_tune_title_desc_match(n_samples, pre_checkpoint, training_seed, name_tag=""):
    """Loads dataset and starts training title-description match classifier model. 

    Args:
        n_samples (int): Not used because a custom size dataset is loaded
        pre_checkpoint (str): The base model used for fine-tuning
        training_seed (int): Seed to keep model head parameters consistent
        name_tag (str, optional): Tag to put at the end of name where model is saved. Defaults to "".
    """
    dataset = load_from_disk(LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME)
    
    # select n_samples with evenly distributed labels
    dataset["train"] = dataset["train"].select(range(n_samples*2))
    positive_and_negative = [dataset["train"].filter(lambda example: example["label"] == label).select(range(int(n_samples/2))) for label in [1,0]]
    dataset["train"] = concatenate_datasets(positive_and_negative).shuffle(seed=2022)

    label2id_map = {"no_match": 0, "match": 1}
    return fine_tune_sequence_classifier(dataset, save_dir=FINE_TUNED_MODEL_TITLE_DESC_MATCH_DIR_NAME+name_tag, label2id_map=label2id_map, 
                                data_preprocessing = data_preprocessing_title_desc_match, pre_checkpoint=pre_checkpoint, multi_label=False, training_seed=training_seed)

def fine_tune_each_task():
    """Runs each of the fine-tuning tasks and saves them, in case they should be shared.
    """
    model_names = {"GenGPT": "roberta-base",
                   "NewsGPT": "models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2450000 (epoch 3)",
                   "GenNewsGPT": "models/roberta_sciride_news_gen_base/checkpoint-last",
                   "NotPretrained": None}
    
    for model_name in model_names:
        load_data_and_fine_tune_covid(2000, model_names[model_name], 1, name_tag="_"+model_name, sub=True)
        load_data_and_fine_tune_multi(2000, model_names[model_name], 1, name_tag="_"+model_name, sub=True)
        load_data_and_fine_tune_title_desc_match(2000, model_names[model_name], 1, name_tag="_"+model_name)
        load_data_and_fine_tune_NER(model_names[model_name], 1, True, name_tag="_"+model_name)

def main():
    # fine_tune_each_task()
    pass

if __name__ == "__main__":
    main()