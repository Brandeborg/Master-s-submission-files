from datasets import DatasetDict, Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast, PreTrainedModel
from data_preparation import pack_inputs_like_roberta, save_dataset_as_arrow
import os
import torch

from dotenv import load_dotenv
load_dotenv()  

# --- This file contains functions in charge of pre-training models --- #

# --- LOAD ENV CONSTANTS FOR CONSISTENT FILE NAMES --- #
FILTERED_DATASET_NODUPS_DIR_NAME = os.getenv("FILTERED_DATASET_NODUPS_DIR_NAME")
PARTIAL_DATASET_DIR_NAME = os.getenv("PARTIAL_DATASET_DIR_NAME")

ROBERTA_PRETRAIN_INPUT_DIR_NAME = os.getenv("ROBERTA_PRETRAIN_INPUT_DIR_NAME")
GPT_PRETRAIN_INPUT_DIR_NAME = os.getenv("GPT_PRETRAIN_INPUT_DIR_NAME")

PRETRAINED_ROBERTA_MODEL_RAND_BASE_DIR_NAME = os.getenv("PRETRAINED_ROBERTA_MODEL_RAND_BASE_DIR_NAME")
PRETRAINED_ROBERTA_MODEL_GEN_BASE_DIR_NAME = os.getenv("PRETRAINED_ROBERTA_MODEL_GEN_BASE_DIR_NAME")

PRETRAINED_GPT_MODEL_RAND_BASE_DIR_NAME = os.getenv("PRETRAINED_GPT_MODEL_RAND_BASE_DIR_NAME")
PRETRAINED_GPT_MODEL_GEN_BASE_DIR_NAME = os.getenv("PRETRAINED_GPT_MODEL_GEN_BASE_DIR_NAME")

def train_tokenizer(dataset: Dataset, save_dir: str, checkpoint="roberta-base"):
    """Given a corpus, train a new tokenizer (with new vocab) using the same algorithm as "checkpoint"

    Args:
        dataset (Dataset): Dataset from which the vocab will be exracted
        save_dir (str): directory name where tokenizer should be saved
        checkpoint (str, optional): A model checkpoint from which a tokenizer will be loaded. Defaults to "roberta-base".
    """
    from transformers import AutoTokenizer

    def get_training_corpus():
        """A function that returns a Python generator of the dataset articles, to avoid loading everything into memory at once.
        The generator is wrapped in a function, so it can be used multiple times.

        Returns:
            Generator: A generator yielding 1000 samples of articles at a time
        """
        return (
            [". ".join([title, description]) for title, description in zip(dataset[i : i + 1000]["title"], dataset[i : i + 1000]["description"])]
            for i in range(0, len(dataset), 1000)
        )

    training_corpus = get_training_corpus()

    # load an existing tokenizer to automatically init all the desired configs of the tokenizer we want to mimic, 
    # such as tokenization algorithm and special tokens
    # It could be loaded manually as well. roberta uses ByteLevelBPETokenizer and has special characters ["<s>","<pad>","</s>","<unk>","<mask>"]
    old_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # pass the article generator and start training new tokenizer with a new vocab
    new_tokenizer: PreTrainedTokenizerFast = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=50265) 

    # save the tokenizer
    new_tokenizer.save_pretrained(save_dir + "_tokenizer") 

def data_preprocessing(raw_datasets: Dataset, tokenizer):
    """Tokenizes dataset. Truncates if sequence is too long, but it won't be if pack_inputs_like_roberta() was run prior.

    Args:
        raw_datasets (Dataset): Dataset of raw text to be tokenized
        tokenizer (Tokenizer): The tokenizer
    """
    def tokenize_function(example):
        return tokenizer(example["article"], truncation=True, max_length=512)
    tokenized_datasets: Dataset = raw_datasets.map(tokenize_function, batched=True, remove_columns=["article", "meta", "new_article_id"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

# !! This function was heavily inspired by a comment on the huggingface forum which refered to the evaluation of the original RoBERTa model
# but I can't find the original source.!!
def mlm_eval(eval_pred):
    """Accuracy calculation for MLM. Compares predicted token to label token for all masked tokens.
    Assuming metric.compute() automatically deduces that (232 == 421) == False and count it as FN/FP 
    as the labels are not binary and it still works.

    Args:
        eval_pred (list): contains batch of predictions (arg-maxed) and labels

    Returns:
        dict: Accuracy score
    """
    from evaluate import load
    metric = load("accuracy")

    labels = eval_pred.label_ids
    predictions = eval_pred.predictions

    del eval_pred

    # create list containing lists of indices pointing to the masked tokens in each label-article in the batch (same indeces of the predictions we are interested in)
    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

    # use the list of index-lists to create a list of the actual label tokens 
    # "labels" contains numpy arrays, where multiple items can be accessed at once 
    # by using lists of indices i.e. array([1,2,3,4,5,6])[[0,1,3]] == array([1,2,4])
    labels = [labels[row][indices[row]] for row in range(len(labels))]

    # flatten the list containing lists of label tokens, so all masked tokens are in a single 1-D list
    labels = [item for sublist in labels for item in sublist]

    # same procedure as for labels using the indices extracted from labels
    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    del indices
    predictions = [item for sublist in predictions for item in sublist]

    # free memory on GPU
    torch.cuda.empty_cache()

    return metric.compute(predictions=predictions, references=labels)

def train_language_model(datasets: DatasetDict, tokenizer: PreTrainedTokenizerFast, base_model: PreTrainedModel, output_dir: str, mlm: bool, resume_from_checkpoint=False):
    """Performs language modeling on a model

    Args:
        datasets (DatasetDict): The raw dataset to be used for training. Should contain a train and validation split.
        tokenizer (PretrainedTokenizerFast): The tokenizer to be used to preprocess Dataset. Will be saved with the final model
        base_model (PreTrainedModel): The base model that will be trained. Can either have random weights or be pretrained
        output_dir (str): The name of the directory, where the model checkpoints will be save
        mlm (bool, optional): Whether or not the languge modeling performed is MASKED language modeling. If False, will use CLM.
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    # only valid for mlm
    def compute_metrics(eval_pred):
        return mlm_eval(eval_pred) if mlm else {}
        
    def preprocess_logits_for_metrics(logits, labels):
        """
        This is a workaround to avoid caching too many tensors that are not needed for computing metrics.
        The idea is to reduce the (sequence_length, vocab_size) size logits matrix to a single (sequence_length, 1) size logits matrix, 
        since we only need one scalar (the prediction) for each sequence token to compare it to the label. This function is called before
        anything is cached in compute_metrics, so we avoid caching ~50000 numbers (vocab size) for each sequence token.
        """
        # argmax returns the index of the value with the highest value, so we must assume the vector indicies corrospond 
        # to token/input ids of the vocabulary. Argmax() is the equivalent of performing max() and taking "indices".
        # The values from which the max is found is the predicted scores (not percentages unless softmax is applied) of each
        # token label.
        # dim(ension) is set to -1 to indicate we want to reduce along the "last" axis, meaninng the innermost list
        pred_ids = torch.argmax(logits, dim=-1)
        del logits
        
        # Whatever is returned here will be in the "predictions" attribute of the eval_pred parameter in compute_metrics(eval_pred)
        # There is no need to pass along the labels, as they will be in the "label_ids" attribute no matter what (provided the dataset is labeled)
        return pred_ids

    tokenized_datasets = data_preprocessing(
        datasets, tokenizer)
    
    # in charge of creating input batches for the model. It makes sure batch samples are equal length (padded) and masks some tokens for mlm.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm, mlm_probability=0.15)

    # gradient accumulation can speed up training when using small batches by accumulating gradients and calculating the parameter updates after n steps
    # we are limited by memory to a batch size of < 8, so gradient accumulation is a way to not take too bad of a hit in training performance, while still decreasing
    # batch size to the required amount
    # Another option to decrease use of memory, is to limit max input size
    training_args = TrainingArguments(
        output_dir=output_dir + "/checkpoints",
        overwrite_output_dir=False,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10_000,
        warmup_steps=50000, #~50000 according to roberta (6% of steps), not required when fine-tuning but used anyway with the intuition, that the already learned weights should not be "shocked"
        learning_rate= 2e-6, # same learning rate as bert (2e-5), 10 times smaller when fine-tuning
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        num_train_epochs=6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4
    )

    # init Trainer instance
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir + "/checkpoint-last")

    del trainer
    del base_model
    torch.cuda.empty_cache()

### Defining the different combinations of language modelling ###
def fine_tune_gpt_on_outlet(dataset_path: str, outlet_name = "cnbc.com", train_size: int=None):
    """Extract articles from given outlet and fine-tune gpt using that data.

    Args:
        dataset_path (str): Path to full dataset
        outlet_name (str, optional): Name of outlet.
        train_size (int, optional): Size of train split used for training. Mostly for testing. Defaults to None.
    """
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from datasets import load_from_disk

    def filter_dataset(example):
        """Return only articles from specfic outlet
        """
        return example["meta"]["outlet"] == outlet_name

    # load dataset with expected format: {title: str, description: str, meta: {outlet: str}}
    # meta can include other features but does not have to
    dataset = load_from_disk(dataset_path)

    print("--- FILTERING DATASET TO GET ARTICLES FROM OUTLET ---")
    outlet_dataset = dataset.filter(filter_dataset)

    print("--- FORMATTING FILTERED DATASET TO FIT INTO THE MODEL ---")
    model_checkpoint = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # doesn't have pad token, so reuse other special token
    tokenizer.pad_token = tokenizer.eos_token 

    # merging title and description into single "article" column
    # read pack_inputs_like_roberta() doc string for more details
    outlet_dataset_packed = pack_inputs_like_roberta(outlet_dataset, tokenizer)

    # save packed dataset, splitting up into train-test-validation sets in the process
    dataset_dir = "dataset_" + outlet_name
    save_dataset_as_arrow(outlet_dataset_packed, 
                          dir=dataset_dir, 
                          shuffle=True, 
                          split=True, 
                          split_size=0.99)

    # load the split dataset
    split_dataset = load_from_disk(dataset_dir)

    # determine amount of training data
    if train_size != None:
        range = min(len(split_dataset["train"]), train_size)
        split_dataset["train"] = split_dataset["train"].select(range(range))

    print("--- FINE-TUNING " + model_checkpoint + " ON ARTICLES FROM " + outlet_name + " ---")
    # load pretrained model and start fine-tuning
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    train_language_model(split_dataset, 
                   tokenizer=tokenizer, 
                   base_model = model, 
                   output_dir = model_checkpoint + "_" + outlet_name, 
                   mlm = False, 
                   resume_from_checkpoint=False)
    
def pretrain_gpt2_news():
    """Load the data for pre-training a GPT-2 architecture from scracth and run train_language_model() using CLM.
    """
    from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer

    dataset = load_from_disk(GPT_PRETRAIN_INPUT_DIR_NAME)

    model_checkpoint = PRETRAINED_GPT_MODEL_RAND_BASE_DIR_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint + "_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
        
    config = GPT2Config(vocab_size=len(tokenizer), 
                        n_ctx=512, 
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,)
    model = GPT2LMHeadModel(config)

    train_language_model(dataset, 
                         tokenizer=tokenizer, 
                         base_model=model, 
                         output_dir=model_checkpoint, 
                         mlm=False, 
                         resume_from_checkpoint=False)

def pretrain_roberta_news():
    """Load the data for pre-training a RoBERTa architecture from scracth and run train_language_model() using MLM.
    """
    from transformers import RobertaForMaskedLM, RobertaConfig, AutoTokenizer

    dataset = load_from_disk(ROBERTA_PRETRAIN_INPUT_DIR_NAME)

    model_checkpoint = PRETRAINED_ROBERTA_MODEL_RAND_BASE_DIR_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint + "_tokenizer")

    config = RobertaConfig(vocab_size=len(tokenizer),
                               max_position_embeddings=514, 
                               bos_token_id=tokenizer.bos_token_id,
                               eos_token_id=tokenizer.eos_token_id,)
    model = RobertaForMaskedLM(config)

    train_language_model(dataset, 
                         tokenizer=tokenizer, 
                         base_model=model, 
                         output_dir=model_checkpoint, 
                         mlm=True, 
                         resume_from_checkpoint=True)

def fine_tune_gpt2_on_news():
    """Load the data for fine-tuning a GPT-2 architecture from an already pre-trained checkpoint and run train_language_model() using CLM.
    """
    from transformers import AutoTokenizer, GPT2LMHeadModel

    dataset = load_from_disk(GPT_PRETRAIN_INPUT_DIR_NAME)

    model_checkpoint = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

    train_language_model(dataset, 
                         tokenizer=tokenizer, 
                         base_model=model, 
                         output_dir=PRETRAINED_GPT_MODEL_GEN_BASE_DIR_NAME, 
                         mlm=False, 
                         resume_from_checkpoint=True)

def fine_tune_roberta_on_news():
    """Load the data for fine-tuning a RoBERTa architecture from an already pre-trained checkpoint and run train_language_model() using MLM.
    """
    from transformers import AutoTokenizer, RobertaForMaskedLM

    dataset = load_from_disk(ROBERTA_PRETRAIN_INPUT_DIR_NAME)

    model_checkpoint = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model = RobertaForMaskedLM.from_pretrained(model_checkpoint)

    train_language_model(dataset, 
                         tokenizer=tokenizer, 
                         base_model=model, 
                         output_dir=PRETRAINED_ROBERTA_MODEL_GEN_BASE_DIR_NAME, 
                         mlm=True,
                         resume_from_checkpoint=False)


def main():
    """Prepare the data and run the training combinations
    """
    from datasets import load_from_disk, Dataset
    import time

    start = time.time()

    ### GPT ###
    # train GPT tokenizer
    from transformers import AutoTokenizer
    article_dataset_full = load_from_disk(FILTERED_DATASET_NODUPS_DIR_NAME)
    train_tokenizer(article_dataset_full, save_dir=PRETRAINED_GPT_MODEL_RAND_BASE_DIR_NAME, checkpoint="gpt2")
    tokenizer_gpt = AutoTokenizer.from_pretrained(PRETRAINED_GPT_MODEL_RAND_BASE_DIR_NAME+"_tokenizer")

    # Package dataset for GPT so it is ready for pre-training
    article_dataset: Dataset = load_from_disk(FILTERED_DATASET_NODUPS_DIR_NAME)
    article_dataset_512 = pack_inputs_like_roberta(article_dataset, tokenizer=tokenizer_gpt)
    save_dataset_as_arrow(article_dataset_512, dir=GPT_PRETRAIN_INPUT_DIR_NAME, shuffle=True, split=True, split_size=0.995)

    # Train
    pretrain_gpt2_news()
    fine_tune_gpt2_on_news()

    ### ROBERTA ###
    # train Roberta tokenizer
    from transformers import AutoTokenizer
    article_dataset_partial = load_from_disk(PARTIAL_DATASET_DIR_NAME)
    train_tokenizer(article_dataset_partial, save_dir=PRETRAINED_ROBERTA_MODEL_RAND_BASE_DIR_NAME, checkpoint="roberta-base")
    tokenizer_roberta = AutoTokenizer.from_pretrained(PRETRAINED_ROBERTA_MODEL_RAND_BASE_DIR_NAME+"_tokenizer")

    # Package dataset for Roberta so it is ready for pre-training
    article_dataset: Dataset = load_from_disk(PARTIAL_DATASET_DIR_NAME)
    article_dataset_512 = pack_inputs_like_roberta(article_dataset, tokenizer=tokenizer_roberta)
    save_dataset_as_arrow(article_dataset_512, dir=ROBERTA_PRETRAIN_INPUT_DIR_NAME, shuffle=True, split=True, split_size=0.995)

    # Train
    pretrain_roberta_news()
    fine_tune_roberta_on_news()


    print("Time: ", time.time() - start, " SEK")


if __name__ == "__main__":
    main()