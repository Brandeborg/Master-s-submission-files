import torch
import os
import evaluate
from transformers import AutoModel
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, concatenate_datasets
from data_preparation import load_json, generalize_NER_dataset, substitute_keywords_in_rows
from data_preprocessing_finetuning import data_preprocessing_seq_class, data_preprocessing_NER, data_preprocessing_title_desc_match

from dotenv import load_dotenv
load_dotenv()  

# --- This file contains all the functions used for evaluation, called either during training, to report other metrics than validation loss, --- #
# --- or after training to evaluate the final models.                                                                                        --- #

# --- LOAD ENV CONSTANTS FOR CONSISTENT FILE NAMES --- #
FINE_TUNED_MODEL_COVID_DIR_NAME = os.getenv("FINE_TUNED_MODEL_COVID_DIR_NAME")
FINE_TUNED_MODEL_POLIT_DIR_NAME = os.getenv("FINE_TUNED_MODEL_POLIT_DIR_NAME")
FINE_TUNED_MODEL_NER_DIR_NAME = os.getenv("FINE_TUNED_MODEL_NER_DIR_NAME")
FINE_TUNED_MODEL_TITLE_DESC_MATCH_DIR_NAME = os.getenv("FINE_TUNED_MODEL_TITLE_DESC_MATCH_DIR_NAME")

LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME = os.getenv("LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME")
LABELED_COVID_CLASS_DATASET_DIR_NAME = os.getenv("LABELED_COVID_CLASS_DATASET_DIR_NAME") 
LABELED_NER_DATASET_DIR_NAME = os.getenv("LABELED_NER_DATASET_DIR_NAME")
LABELED_NER_DATASET_NAMESPLIT_DIR_NAME = os.getenv("LABELED_NER_DATASET_NAMESPLIT_DIR_NAME")
LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME = os.getenv("LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME")

# global value deciding how many test samples are extracted from the passed test splits
test_size = 200

def predict_labels(model: AutoModel, eval_dataloader: DataLoader, handle_logits, id2label_map: dict = None):
    """Given a model and some evaluation data via a data_loader, run the data through the model to calculated the predictions.

    Args:
        model (AutoModel): The model used for evaluation
        eval_dataloader (DataLoader): Data to be evaluated.
        handle_logits (_type_): A callback to handle the logits coming out of the model. Logits are handled differently for multi label data for instance.
        id2label_map (dict, optional): Optional dict functioning as a map from label id to label string. Defaults to None.

    Returns:
        (tuple): list of predictions, list of references/true labels
    """
    # clear cache before using the GPU to avoid memory crash
    torch.cuda.empty_cache()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    all_predictions = []
    all_references  = []

    # set to eval mode
    model.to(device=device)
    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

            # if id2label map is avaible pass it to the logits handler
            predictions, references = handle_logits(outputs.logits, batch["labels"]) if id2label_map == None else handle_logits(outputs.logits, batch["labels"], id2label_map)
            all_predictions.extend(predictions)
            all_references.extend(references)
            del outputs
            del predictions
            del references

    # set to train mode
    model.train()
    del model

    return all_predictions, all_references

def handle_logits_NER(logits, labels, id2label):
    """Apply argmax to logits and call handle_predictions_NER(). handle_logits and handle_predictions are split up because handle_predictions cannot be performed
    during preprocess_logits_for_metrics. handle_predictions_NER returns strings and preprocess_logits_for_metrics returns a tensor, that cannot contain strings. 
    So, during fine_tuning, the torch.argmax() function is performed during preprocessing and then the resulting predictions are passed to compute_metrics which 
    calls handle_predictions_NER.
    During evaluation AFTER finetuning, handle_logits_NER is run, thus performing both at the same time.

    Args:
        predictions (list): The prediction logits (lists of scores)
        labels (list): The actul label ids
        id2label (dict): a map from label id to label text
    """
    predictions = torch.argmax(logits, dim=-1)
    return handle_predictions_NER(predictions, labels, id2label)

def handle_predictions_NER(predictions, labels, id2label):
    """Exchange label and prediction ids with their corrospoding text label, because seqeval only takes strings.
    Also, remove -100 from labels and predictions, because they are meant to be ignored.
    Note that the prediction wont actually be -100, but the predicictions with the same index as a "-100 label" is removed.

    Args:
        predictions (list): The predicted label ids
        labels (list): The actul label ids
        id2label (dict): a map from label id to label text

    Returns:
        tuple: actual labels and label predictions as strings
    """
    
    true_predictions = [
        [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [[id2label[int(l)] for l in label if l != -100] for label in labels]

    return (true_predictions, true_labels)

def handle_logits_single_label(logits, labels):
    """Simply applies argmax to output

    Returns:
        tuple: Prediction_labels, actual_labels
    """
    predicted_labels = torch.argmax(logits, dim=-1)
    return predicted_labels, labels

def handle_logits_multi_label(logits, labels):
    """Handle the logits for multi label data. It is not enough to just take argmax as multiple labels is possible.
    Instead, take the ones above a certain threshold.

    Args:
        logits (list): logits resulting from passing text through seq class model
        labels (list): The actual labels.

    Returns:
        _type_: _description_
    """
    # NOTE: For multi_label classification, the f1 scores is much more interesting than accuracy, due to how we label the data. 
    # The majority of labels will be 0, since there is a binary label for each possible label for each sentence.
    # Due to predicting the label based on a threshhold, the predictions will also primarily be 0. That means,
    # there will be a huge majority of true negatives, which increases accuracy but not f1, since f1 does not account for TN
    
    # threshold above which a predicted label's likelyhood should be.
    threshold = 0.33

    # using sigmoid to have the scores be between 0.0 and 1.0 for each label prediction
    sigmoid_predictions = torch.sigmoid(logits)

    # converting the predictions to binary integers by having every prediction above a certain threshold be 1 for true, and 0 (false) for everything below.
    binary_predictions = []
    for prediction in sigmoid_predictions:
        binary_prediction = [1 if label_score > threshold else 0 for label_score in prediction]
        binary_predictions.append(binary_prediction)

    del sigmoid_predictions

    # flattening the batch of label and prediction lists because the metric function can't handle list of lists
    references = torch.flatten(labels.int())
    predictions = torch.flatten(torch.tensor(binary_predictions))

    return predictions, references

def evaluate_NER_classification(raw_dataset, fine_checkpoint, id2label_map):
    """Given a dataset of raw text with labels, evaulate the model loaded from the checkpoint

    Args:
        raw_dataset (Dataset): Dataset of lists of raw words and token labels (NE)
        fine_checkpoint (str): The model checkpoint to load
        id2label_map (dict): Map from label id to label string

    Returns:
        dict: Evaluation results
    """
    from transformers import RobertaForTokenClassification
    model = RobertaForTokenClassification.from_pretrained(fine_checkpoint)

    # run preprocessing, tokenizing the input text into input_ids
    _, tokenized_dataset, data_collator = data_preprocessing_NER(raw_dataset=raw_dataset, checkpoint=fine_checkpoint)

    eval_dataloader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=data_collator)

    predicted_labels, actual_labels = predict_labels(model, eval_dataloader=eval_dataloader, handle_logits=handle_logits_NER, id2label_map=id2label_map)

    del model

    seqeval = evaluate.load("seqeval")
    return seqeval.compute(predictions=predicted_labels, references=actual_labels)

def evaluate_seq_classification(model, raw_dataset, data_preprocessing, tokenizer, multi_label):
    """Given a model and a dataset of raw text, evaluate the models predictiona accuracy on the data.

    Args:
        model (PretrainedModel): A fine-tuned sequence classification model.
        raw_dataset (Dataset): A labeled sequence classification dataset
        data_preprocessing (Callable): A callback for pre_processing the data (tokenize)
        tokenizer (Tokenizer): A tokenizer passed data_preprocessing. 
        multi_label (bool): Whether the raw_dataset labels are multi-label

    Returns:
        _type_: _description_
    """
    from transformers import DataCollatorWithPadding
    import evaluate

    # run preprocessing, tokenizing the input text into input_ids
    tokenized_dataset = data_preprocessing(raw_dataset, tokenizer=tokenizer, multi_label=multi_label)
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=data_collator)

    # load the metrics used to evaluate seq classification
    metric = evaluate.combine([evaluate.load("accuracy"), evaluate.load("f1")])

    # choose a function to handle logits based on "multi_label"
    handle_logits = handle_logits_multi_label if multi_label else handle_logits_single_label

    # get the formatted results predicted by the model
    predictions, references = predict_labels(model, eval_dataloader, handle_logits)

    del model

    # evaluate the results
    metrics = metric.compute(predictions=predictions, references=references)
    del metric
    return metrics

# wrapper functions
### --- NER --- ###
def load_data_and_evaluate_NER(gen=True):
    """Load a NER dataset to be evaluated and evaluate it.

    Args:
        gen (bool, optional): Whether the NER data should be "generalized" (run generalize_NER_dataset()). Defaults to True.

    Returns:
        dict: The evaluation results
    """
    ner_dataset_test: Dataset = load_from_disk(LABELED_NER_DATASET_NAMESPLIT_DIR_NAME)["test"]

    if (gen):
        ner_dataset_test = generalize_NER_dataset(ner_dataset_test)

        label2id = {"O": 0, "B-NE": 1, "I-NE": 2}
        id2label = {v: k for k, v in label2id.items()}
    else:
        label2id = load_json("blobs/NER_labelmap.json")
        id2label = {v: k for k, v in label2id.items()}

    return evaluate_NER_classification(raw_dataset=ner_dataset_test, fine_checkpoint=FINE_TUNED_MODEL_NER_DIR_NAME + "/checkpoint-last", id2label_map=id2label)

### --- COVID --- ###
def load_data_and_evaluate_covid(sub=False):
    """Load a single-label (COVID) dataset to be evaluated and evaluate it.

    Args:
        sub (bool, optional): Whether the COVID keywords should be substituted with <mask> (run substitute_keywords_in_rows()). Defaults to False.

    Returns:
        dict: The evaluation results
    """
    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
    
    fine_checkpoint = FINE_TUNED_MODEL_COVID_DIR_NAME + "/checkpoint-last"
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(fine_checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(fine_checkpoint)

    covid_dataset_test: Dataset = load_from_disk(LABELED_COVID_CLASS_DATASET_DIR_NAME)["test"]

    # select n_samples with evenly distributed labels
    covid_dataset_test = covid_dataset_test.select(range(test_size*2))
    positive_and_negative = [covid_dataset_test.filter(lambda example: example["label"] == label).select(range(int(test_size/2))) for label in [1,0]]
    covid_dataset_test = concatenate_datasets(positive_and_negative).shuffle(seed=2022)

    if (sub):
        covid_class_keywords = load_json("blobs/covid_class_keywords.json")
        keywords = [keyword for cls in covid_class_keywords.values() for keyword in cls["keywords"]]
        covid_dataset_test = substitute_keywords_in_rows(covid_dataset_test, keywords, "<mask>")

    return evaluate_seq_classification(model, raw_dataset=covid_dataset_test, data_preprocessing=data_preprocessing_seq_class, tokenizer=tokenizer, multi_label=False)

### --- MULTI --- ###
def load_data_and_evaluate_multi(sub=False):
    """Load a multi-label (politicians) dataset to be evaluated and evaluate it.

    Args:
        sub (bool, optional): Whether the politician keywords should be substituted with <mask> (run substitute_keywords_in_rows()). Defaults to False.

    Returns:
        dict: The evaluation results
    """
    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
    
    fine_checkpoint = FINE_TUNED_MODEL_POLIT_DIR_NAME + "/checkpoint-last"
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(fine_checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(fine_checkpoint)

    # load the specially created dataset with even distribution of labels
    multi_dataset_test: Dataset = load_from_disk(LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME)["test"]

    if (sub):
        multi_class_keywords = load_json("blobs/multi_class_keywords.json")
        keywords = [keyword for cls in multi_class_keywords.values() for keyword in cls["keywords"]]
        multi_dataset_test = substitute_keywords_in_rows(multi_dataset_test, keywords, "<mask>")
    
    return evaluate_seq_classification(model, raw_dataset=multi_dataset_test, data_preprocessing=data_preprocessing_seq_class, tokenizer=tokenizer, multi_label=True)

### --- TITLE DESC MATCH --- ###
def load_data_and_evaluate_title_desc_match():
    """Load a single-label (title-desc match) dataset to be evaluated and evaluate it.

    Returns:
        dict: The evaluation results
    """
    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
    
    fine_checkpoint = FINE_TUNED_MODEL_TITLE_DESC_MATCH_DIR_NAME + "/checkpoint-last"
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(fine_checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(fine_checkpoint)

    title_desc_dataset_test: Dataset = load_from_disk(LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME)["test"]

    # select n_samples with evenly distributed labels
    title_desc_dataset_test = title_desc_dataset_test.select(range(test_size*2))
    positive_and_negative = [title_desc_dataset_test.filter(lambda example: example["label"] == label).select(range(int(test_size/2))) for label in [1,0]]
    title_desc_dataset_test= concatenate_datasets(positive_and_negative).shuffle(seed=2022)
    
    return evaluate_seq_classification(model, raw_dataset=title_desc_dataset_test, data_preprocessing=data_preprocessing_title_desc_match, tokenizer=tokenizer, multi_label=False)