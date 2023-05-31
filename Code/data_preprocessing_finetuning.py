from datasets import Dataset
from transformers import PreTrainedTokenizerFast

# --- This file contains the pre-processing functions used immediately before training or evaluating, performing tokenization which data_preparation does not do. --- #

def data_preprocessing_seq_class(raw_dataset: Dataset, tokenizer: PreTrainedTokenizerFast, multi_label: bool, columns: list = ["title", "description"]):
    """Takes a labeled sequence classification dataset and tokenizes it using the passed tokenizer.
    It truncates and packs the overflowing tokens in a new row. If "pack_inputs_like_roberta()" is used before, truncation has no effect.

    Args:
        raw_dataset (Dataset): Dataset of raw text
        tokenizer (PreTrainedTokenizerFast): A tokenizer use to tokenize data
        multi_label (bool): Whether or not the data is multi labeled. If it is "reformat_labels()" is used.
        columns (list, optional): Which columns of the dataset includes raw text and should be tokenized. Defaults to ["title", "description"].

    Returns:
        Dataset: A with "columns" removed an tokenized "input_ids" as a replacement.
    """
    from datasets import Sequence, Value

    # renaming the "label" column in single_label datasets because the model always needs the column to be called "labels"
    if not multi_label: raw_dataset = raw_dataset.rename_column("label", "labels")

    # exrtacting the class_names early because the labels column will need to be converted from [ClassLabels] to [Floats],
    # and thus the class names are lost. (Only applies to multi_label)
    if multi_label: class_names = raw_dataset.features["labels"].feature.names

    def tokenize_function(examples):
        """Tokenizing a single row. Passed to ".map()".

        Args:
            examples (dict): A row of data

        Returns:
            dict: Tokenized row
        """
        # making list of the columns specified by the "columns" parameter
        text_columns = [examples[column] for column in columns]

        # zipping the text columns to create pairs of (title, description) for instance. Note: unpacking argument becuase zip does not accept list of lists
        text_tuples = list(zip(*text_columns))

        # creating list of the text tuples merged to a single text
        merged_texts = [". ".join(texts) for texts in text_tuples]

        # return_overflowing_tokens = True, meaning new samples are created from the overflow when num_tokens > 512 (max_input for roberta)
        result = tokenizer(merged_texts, truncation=True, return_overflowing_tokens=True)

        # since the input_ids-column (tokenized_input) created by the tokenizer is now potentially longer than the labels column,
        # it needs to be fixed. It can be done by copying the label from the original samples to the overflow samples.
        # This needs to be done for labels especially, because we still need a label for each input, but it also needs to be done for all other columns,
        # becuase the dimenstions of the dataset/matrix columns need to match.

        # overflow_to_sample_mapping returns an expanded list of indices, each element in the expanded list pointing to an index in the original list 
        # Example: 
        # original list: ["sample1", "sample2"]. 
        # "sample2" is 512+ tokens, so it overflows and creates a new sample. 
        # expanded list: ["sample1", "sample2", "sample3"] 
        # overflow_to_sample_mapping: [0,1,1]
        overflow2sample_map = result.pop("overflow_to_sample_mapping")

        # adding the original data to each column of the expanded sample list
        for key, values in examples.items():
            result[key] = [values[i] for i in overflow2sample_map]
        return result
    
    def reformat_labels(example):
        """Reformatting label lists to be list of true/false values (as floats), like this: [0,3,4] -> [1.0, 0.0, 0.0, 1.0, 1.0],
        meaning a sequence label with the ids [0,3,4], now has its labels represented by having the index corrosponding to the 
        label id be 1.0 if it has the label an vice versa. It needs to be floats because the model needs it to be floats, 
        and it does not convert them itself.
        """
        label_bools = [0.0] * len(class_names)

        for label in example["labels"]:
            # the labels are already floats at this point, but all of them are whole numbers and can still be used as indices
            label_bools[int(label)] = 1.0
        
        return {"labels": label_bools}
    
    def adjust_attention_mask(example):
        """If the special mask character of the vocab appears in the input_ids, 
        set the corrosponding attention_mask index to 0.

        Args:
            example (dict): A dataset row with "input_ids" and "attention_mask" columns.

        Returns:
            dict: dict with new attention mask
        """
        mask_id = tokenizer.mask_token_id
        new_attention_mask = [0 if token_id == mask_id else 1 for token_id in example["input_ids"]]
        return {"attention_mask": new_attention_mask}


    if multi_label: raw_dataset = raw_dataset.cast_column("labels", Sequence(Value("float32")))
    if multi_label: raw_dataset = raw_dataset.map(reformat_labels)
    tokenized_dataset: Dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_dataset: Dataset = tokenized_dataset.map(adjust_attention_mask)
    tokenized_dataset = tokenized_dataset.remove_columns(columns+["meta", "new_article_id"])
    
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

def data_preprocessing_NER(raw_dataset: Dataset, checkpoint):
    """Creates the items needed before starting training NER model, such as a tokenizer, a tokenized dataset and data collator.

    Args:
        raw_dataset (Dataset): Dataset with lists of "raw words" and corrosponding token labels
        checkpoint (str, optional): _description_. Defaults to "roberta-base".

    Returns:
        tuple: (tokenizer, tokenized_datasets, data_collator)
    """
    from transformers import AutoTokenizer, DataCollatorForTokenClassification

    # add_prefix_space = True, meaning the first word will be treated the same as all other words that get a "Ġ" in front, due to the space in front of them.
    # Since the sequences are pretokenized (split up into words), it is also a requirement from RobertaTokenizer, otherwise none of the words would get a Ġ.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)

    # inspired by HF token classification tutorial
    # https://huggingface.co/docs/transformers/tasks/token_classification
    def tokenize_and_extend_labels(examples):
        """
        BPE example:
        ['This', 'is', 'Buffo.']
        [   0,     0,     1    ]
                |
                V
        ['ĠThis', 'Ġis', 'ĠBuff', 'o', '.']
        [   0,      0,      1,   -100,-100 ]

        Args:
            examples (dict): dict with list of words and labels
        """
        tokenized_sequences = tokenizer(examples["words"], is_split_into_words=True, truncation=True)

        all_labels = []

        # The sequence tokens have been spread out to no longer lign up with the label list associated with each sequence
        for i, labels in enumerate(examples["labels"]):
            # get a token2word mapping for the sequence at index i in the batch.
            # Example: if ["It", "is", "raining"] was tokonized to [CLS, "It", "is", "rain", "##ing", SEP],
            # word_ids is [None, 0, 1, 2, 2, None]
            # Note that the lists being letters is for example's sake. They are really numbers (input ids).
            word_ids = tokenized_sequences.word_ids(batch_index=i)

            previous_word_id = None
            label_ids = []

            # loop through the sequence's word_ids (one for each token) and create new list of token labels
            for word_id in word_ids: 
                if word_id is None:
                    # input_ids created by the tokenizer (CLS, SEP, etc.) should be ignored by the model
                    label_ids.append(-100)
                # !! reconsider this. Maybe dont ignore part of word, because it could mean "(ABC News)" would only get "Ġ(" tagged as the NE !!
                # if the current token is the beginning of a word ("rain" in the example), give it the original label 
                elif word_id != previous_word_id: 
                    label_ids.append(labels[word_id])
                # otherwise ignore it
                else:
                    label_ids.append(-100)
                previous_word_id = word_id

            # add the expanded lable list to the batch of label lists
            all_labels.append(label_ids)

        tokenized_sequences["labels"] = all_labels
        return tokenized_sequences

    tokenized_datasets = raw_dataset.map(tokenize_and_extend_labels, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["words","meta"])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    return (tokenizer, tokenized_datasets, data_collator)

def data_preprocessing_title_desc_match(labeled_datasets: Dataset, tokenizer, multi_label = False):
    """The most basic data preprocessing. Simply tokenizes raw dataset and removes unneeded columns. Truncates if samples are too long. See data_preprocessing_seq_class
    for more complicated implementation. 

    Args:
        labeled_datasets (Dataset): dataset with title, description and labels indicating if title and description matches.
        tokenizer (RobertaTokenizerFast): Tokenizer used to convert raw text to input ids
        multi_label (bool) = DON'T REMOVE. Needed because the function is gonna be passed as callback to a function that calls it with the assumption that it has it.
    """
    def tokenize_function(example):
        # Passing two seqeunces to roberta tokenizer simply results in [SEP] token between them. In the case of roberta. sep == eos.
        # So really, the result is two [SEP] between title and description.
        return tokenizer(example["title"], example["description"], truncation=True)
    
    tokenized_dataset = labeled_datasets.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["title","description"])

    return tokenized_dataset