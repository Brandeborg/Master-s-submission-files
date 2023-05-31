import errno
import json
import gzip
import os
from os import getcwd, listdir, mkdir
from os.path import join, abspath
from datasets import Dataset, DatasetDict, load_from_disk, disable_caching
from transformers import PreTrainedTokenizerFast

from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.util import bigrams
import string
import html

# --- This file contains functions meant to prepare the data for training in terms of extraction, cleaning and labeling but not tokenization  --- #
# --- The result is a bunch of HuggingFace datasets for different purposes                                                                    --- #

# --- LOAD ENV CONSTANTS FOR CONSISTENT FILE NAMES --- #
from dotenv import load_dotenv
load_dotenv()  
DATA_FILES_JSON_DIR_NAME = os.getenv("DATA_FILES_JSON_DIR_NAME") 

DATASET_DIR_NAME = os.getenv("DATASET_DIR_NAME")
DATASET_NODATES_DIR_NAME = os.getenv("DATASET_NODATES_DIR_NAME")
DATASET_NODATES_NODUPS_DIR_NAME = os.getenv("DATASET_NODATES_NODUPS_DIR_NAME")
DATASET_NODATES_NODUPS_CLEANED_DIR_NAME = os.getenv("DATASET_NODATES_NODUPS_CLEANED_DIR_NAME")

FILTERED_DATASET_DIR_NAME = os.getenv("FILTERED_DATASET_DIR_NAME")
FILTERED_DATASET_NODUPS_DIR_NAME = os.getenv("FILTERED_DATASET_NODUPS_DIR_NAME")
GPT_PRETRAIN_INPUT_DIR_NAME = os.getenv("GPT_PRETRAIN_INPUT_DIR_NAME")

PARTIAL_DATASET_DIR_NAME = os.getenv("PARTIAL_DATASET_DIR_NAME")
ROBERTA_PRETRAIN_INPUT_DIR_NAME = os.getenv("ROBERTA_PRETRAIN_INPUT_DIR_NAME")

LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME = os.getenv("LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME")
LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_DIR_NAME = os.getenv("LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_DIR_NAME")
LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME = os.getenv("LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME")
LABELED_COVID_CLASS_DATASET_DIR_NAME = os.getenv("LABELED_COVID_CLASS_DATASET_DIR_NAME") 

LABELED_NER_JSON_FILE_NAME = os.getenv("LABELED_NER_JSON_FILE_NAME")
LABELED_NER_DATASET_DIR_NAME = os.getenv("LABELED_NER_DATASET_DIR_NAME")

LABELED_NER_DATASET_NAMESPLIT_DIR_NAME = os.getenv("LABELED_NER_DATASET_NAMESPLIT_DIR_NAME")

LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME = os.getenv("LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME")

PRETRAINED_ROBERTA_MODEL_DIR_NAME = os.getenv("PRETRAINED_ROBERTA_MODEL_DIR_NAME")
PRETRAINED_GPT_MODEL_DIR_NAME = os.getenv("PRETRAINED_GPT_MODEL_DIR_NAME")

# --- FUNCTIONS FOR INITIAL DATA EXTRACTION --- #
def create_english_outlets_json():
    """Create json file (extracted from AvailableOutlets.json) containing all the english outlets as keys. 
    
    Adds info about language, including whether or not they may contain some foreign articles, defaulting to False.
    After the functino has has run, manually go through AvailableOutlets.txt, look at the comments and amend the "some_foreign" fields of english_outlets.json
    """
    with open("./blobs/AvailableOutlets.json") as json_file:
        english_outlets = {}
        outlets = json.load(json_file)

        for outlet, data in outlets.items():
            if ("en" in data["lng"]):
                english_outlets[outlet] = {"language": "en", "some_foreign": False}

    # sort dict by key
    english_outlets = dict(sorted(english_outlets.items(), key=lambda item: item[0]))

    # save the english outlets dict as json
    with open("./blobs/english_outlets.json", "w", encoding='utf-8') as f:
        json_record = json.dumps(english_outlets)
        f.write(json_record)

def load_english_outlets():
    """Load the english_outlets.json file and return as dict.

    Returns:
        dict: A dictionary with all english outlets as keys.
    """
    with open("./blobs/english_outlets.json") as json_file:
        return json.load(json_file) 

def extract_sciride_data_to_json():
    """Traverses the sciride data, extract the necessary fields and saves them in a jsonl format, 
    that can be loaded with the Hugging Face ".load_dataset()" function.
    """
    print("--- MOVING NEEDED DATA FIELDS FROM THE SCIRIDE DATASET TO JSON FILES ---")
    data_dir = "release"
    path_to_outlets = abspath(join(getcwd(), data_dir))
    eng_outlets = load_english_outlets()

    for outlet in listdir(path_to_outlets):
        if outlet not in list(eng_outlets.keys()):
            continue

        # create directory holding the dataset .jsonl-files unless it exists
        try:
            mkdir(DATA_FILES_JSON_DIR_NAME)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # iterate through all days and add the wanted fields to a the outlet's output .jsonl-file
        path_to_days = join(path_to_outlets, outlet, "per_day")
        for day in listdir(path_to_days):
            # unzip articles from day and load the json file as a Python dict
            try:
                path_to_day = join(path_to_days, day)
                day_articles: dict = json.load(gzip.open(path_to_day))
            except:
                continue

            # extract the needed fields from the day's articles and add them to the .jsonl-file in bulk
            articles = []
            for art_id, article in day_articles.items():
                reformatted_article = \
                    {"meta":
                     {
                        "article_id": art_id, "outlet": outlet, "date": day[:-3]
                     },
                        "title": article["title"],
                        "description": article["description"]
                     }
                articles.append(reformatted_article)

            path_to_output_csv = "%s/%s.jsonl" % (DATA_FILES_JSON_DIR_NAME, outlet)
            with open(path_to_output_csv, "a+", encoding='utf-8') as f:
                for line in articles:
                    json_record = json.dumps(line, ensure_ascii=False)
                    f.write(json_record + '\n')
        
        print("finished extracting from: ", outlet)            
    
def load_dataset_from_files():
    """Load the data files created using extract_sciride_data_to_json() as a Huggingface DatasetDict, with a split (Dataset) for each outlet
    
    Returns:
        DatasetDict: The Huggingface datasets loaded from files, split up into outlets, ordered by date
    """
    from datasets import load_dataset
    
    datasets = DatasetDict()

    progress_count = 0
    english_outlets = list(load_english_outlets())
    for outlet in english_outlets:
        progress_count += 1
        print(progress_count, "/", len(english_outlets))

        # load articles from a single outlet into a dataset
        data_folder = DATA_FILES_JSON_DIR_NAME
        data_files = "%s/%s.jsonl" % (data_folder, outlet)
        outlet_dataset: Dataset = load_dataset("json", data_files=data_files)

        # load_dataset automatically puts the data into a train split. We don't need splits within splits, so it is unwrapped
        # Add the outlet_dataset to the dataset dict
        datasets[outlet] = outlet_dataset["train"]

    return datasets

# --- FUNCTIONS FOR HANDLING DUPLICATES AND EXTRACTION OF SENTENCES
def remove_duplicates_within_splits(datasets: DatasetDict):
    """Creates a dataset of mostly unique rows by going through the datasets in datasets (usecase: each outlet as a separate dataset) and:
    - removing duplicates from each split
    - move it to a new DatasetDict with the same splits.

    Note that this does not remove duplicates across the entire dataset, only within each split, but it was done this way
    due to memory limitations. Tech such as "dask dataframes using parque files" was attempted to reduce memory, but with no luck.
    The custom remove_duplicates() function is used on each script, which was implemented to reduce memory.
     
    It should be fine this way, since duplicate articles across outlets are probably unlikely apart from extremely generic headlines,
    but this function can just be used as the first step, where the second step is to merch the splits and use remove_duplicates() on everything.


    Example of a duplicate article:
    art_id: 78d66dce4c0e768be39cb490e3f76e8e
    outlet: 9news.com.au
    dates: 20170916-20170924

    Args:
        dataset (DatasetDict): A dataset dict, usually split up into outlets.

    Returns:
        DatasetDict: A dataset dict consisting of unique rows wiithin each split.
    """
    from datasets import DatasetDict

    no_dups_datasets = DatasetDict()

    progress_count = 0
    duplicates_removed = 0
    # loop through each dict key in datasets
    for split in datasets:
        progress_count += 1
        print(progress_count, "/", len(datasets))

        # get the current dataset using the split-key
        dataset: Dataset = datasets[split]
        
        # remove duplicates
        rows_before = dataset.num_rows
        filtered_dataset = remove_duplicates(dataset)
        rows_after = filtered_dataset.num_rows

        # add the filtered dataset to the nodups-DatasetDict
        no_dups_datasets[split] = filtered_dataset
        del dataset

        duplicates_removed += rows_before-rows_after
    
    print("Duplicates removed: ", duplicates_removed)

    return no_dups_datasets

def create_dataset_of_sentences(dataset: Dataset, drop_internal_duplicates: bool = False):
    """Takes the title and description from each row of a dataset, extracts the sentences from each column, and adds the sentences to a new dataset.
    Meta data is duplicated and added to the row along with each found sentence.
    
    NOTE: this function was ultimately not used, as the decision was made to pre-train on the whole articles, not individual senteces,
    but it could be useful, so it's still here.

    Args:
        dataset (Dataset): A dataset with title, description and meta columns
        drop_internal_duplicates (bool): Remove duplicate sentences within each article. Defaults to False.
    """
    import hashlib
    
    def extract_sentences(examples):
        all_metas = []
        all_sentences = []
        all_ids = []

        for i in range(len(examples["title"])):
            sentences = []
            # split sequences up into sentences and add them to list
            sentences.extend(sent_tokenize(examples["title"][i], language="english"))
            sentences.extend(sent_tokenize(examples["description"][i], language="english"))

            # creating a dict from the items in the list, will remove duplicates due to the nature of dicts
            if drop_internal_duplicates: sentences = list(dict.fromkeys(sentences))

            # slitting the row up into sentences will add more rows. add equally as many additional meta fields
            metas = [examples["meta"][i] for j in range(len(sentences))]

            # create a unique id for each unique sentence. Used to remove duplicates later
            ids = [hashlib.md5(sentence.encode("utf-8")).hexdigest() for sentence in sentences]

            all_sentences.extend(sentences)
            all_metas.extend(metas)
            all_ids.extend(ids)

        return {"sentence": all_sentences, "meta_temp": all_metas, "sentence_id": all_ids}

    # mapping is batched because it enables returning a list, which in turn enables transforming a single row into multiple
    column_names = dataset.column_names
    sentence_dataset: Dataset = dataset.map(extract_sentences, batched=True, remove_columns=column_names)
    sentence_dataset = sentence_dataset.rename_column("meta_temp", "meta")
    return sentence_dataset

# --- FUNCTIONS FOR CLEANING AND FILTERING ---
def gauge_junk_in_rows(datasets: DatasetDict, columns, delimeter = "|", file_name="junk_found_with_base"):
    """For every sequence in the dataset, count the junk snippets found after the character "|". 
    Saves a json file containing a dict of all the different junk snippets/phrases as keys,
    along with the amount that was found and what came before the first "|", the first time it was found.

    This function was used to create a json file, in which to manually guage the junk snippets, in order to create a better junk finder function.
    For instance to find out what the longest a (repeating) junk snippet usually is, which is used to set a "junk_max_size" in 
    find_junk_in_rows().
    It was also used to see if there was ever any junk BEFORE the first "|", which was the case. ("Opinion | bla bla" for instance.)

    Args:
        datasets (DatasetDict): A dataset of sequences
        columns (list): The columns of the dataset to check for junk in.
        delimeter (str, optional): Character to split sentence on. Defaults to "|".
        file_name (str, optional): Name of file the junk_snippet counts are save to. Do not inlcude file type postfix and folder

    Returns:
        dict: A dict containing counts of all junk found, along with the "base" of the junk snippet. The base is what came before the first "|", 
        which in most cases should be the "real" sentence.
    """
    # if the .map() function determines that a specific dataset.map() has already been run, it will load a cached verison,
    # preventing junk_counts from being filled, as the .map won't run. Caching is therefore disabled
    disable_caching()

    # init dict of junk counts
    junk_counts = {}

    def gauge_junk_in_row(example):
        """Finds and counts junk in a row

        Args:
            example (dict): An article row

        Returns:
            dict: returns example unchanged. The point of the function is the side effect (counting).
        """
        for column in columns:
            sequence_snippets = example[column].split(delimeter)

            # if junk is found, increase its count otherwise add it to the dict with a count of 1, 
            # along with its "first base", meaning what came before the junk
            junk_snippets = sequence_snippets[1:]
            if junk_snippets != []:
                for snippet in junk_snippets:
                    if delimeter + snippet in junk_counts:
                        junk_counts[delimeter + snippet]["count"] += 1
                    else:
                        junk_counts[delimeter + snippet] = {"count": 1, "first_base": sequence_snippets[0]}

        return example

    # run gauge_junk_in_row for every row
    datasets = datasets.map(gauge_junk_in_row)

    # sort dict by value
    junk_counts = dict(sorted(junk_counts.items(), key=lambda item: item[1]["count"]))
    
    # save dict
    with open("./blobs/%s.json" % file_name, "w", encoding='utf-8') as f:
        json_record = json.dumps(junk_counts)
        f.write(json_record)

    return junk_counts

def find_junk_in_rows(datasets: DatasetDict, columns: list, delimeter="|", junk_max_length=98):
    """Used to split sequences on a certain character usually associated with junk snippets, in order to count the occurences 
    of these junk snippets, and create a dictionary of known junk snippets as keys, with the count as values. 
    Only keys with a count of 3 or higher is saved in the end. 
    This dict/json can be used to remove junk snippets from rows,
    according to specified rules:
    For instance "for all sequences, remove all junk-substrings, that appeared in the dataset more than n times."

    Default value of junk_max_length is based on the length of the largest junk key found with gauge_junk_in_rows:
    "| Latest news, breaking stories and comment from the London Evening Standard" - 76 chars
    " - have you been paying attention to what's been going on in the world during the past seven days?" - 98 chars
    A max size is defined to limit the size of the dict by excluding some "real" article snippets

    Args:
        datasets (DatasetDict): A dataset of sequences
        columns (list): The columns of the dataset to check for junk in.
        delimeter (str, optional): Character to split sequence on. Defaults to "|".
        junk_max_length (int, optional): max length of key saved in dict. Defaults to 98. 

    Returns:
        dict: contains all junk keys and their count
    """
    # if the .map() function determines that a specific dataset.map() has already been run, it will load a cached verison,
    # preventing junk_counts from being filled, as the .map won't run. Caching is therefore disabled
    disable_caching()

    # init dict of junk counts
    junk_counts = {}

    def find_junk_in_row(example):
        """Finds and counts junk in a row

        Args:
            example (dict): An article row

        Returns:
            dict: returns example unchanged. The point of the function is the side effect (counting).
        """
        for column in columns:
            if delimeter not in example[column]:
                continue

            # separate a sequnce using the delimter to get a list of potential junk snippets
            sequence_snippets = example[column].split(delimeter)

            for i, snippet in enumerate(sequence_snippets):
                # the junk key needs the delimeter added, to prevent it from being removed from the middle
                # of a sentence. 
                # Example: 
                # "UK" in "The UK has decided to..." is fine but
                # "UK" in "Boris Johnson has said ... | UK" is junk 
                # so "| UK" will be considered junk, not "UK"

                # if the snippet is the first in a sequence, the delimter it added after: "Opinion |"
                # otherwise it is added before: "| UK"
                junk_key = snippet + delimeter if i == 0 else delimeter + snippet

                if len(junk_key) > junk_max_length: continue 
                
                if junk_key in junk_counts:
                    junk_counts[junk_key] += 1
                else:
                    junk_counts[junk_key] = 1

        return example

    datasets = datasets.map(find_junk_in_row)

    # filter dict
    # exlclude everything that only occurs 2 times or less
    junk_counts = {k: v for k, v in junk_counts.items() if v > 2}   

    # sort dict by value
    junk_counts = dict(sorted(junk_counts.items(), key=lambda item: item[1]))

    return junk_counts

def remove_junk_from_rows(dataset: Dataset, columns: list, junk_snippets: dict, only_snippets=False):
    """Remove all junk from the sequences accoring to the "junk_snippets" json as well as some additional detected junk patterns.
    
    A lot of the additional regex patterns are different variations of dates attached at the end of a title. 
    These dates make otherwise identical articles unique, which means they would be glossed over during duplicate removal.
    By removing most dates, additional duplicats can be removed
    Some date formats are kept, such as "- 21st March 2016" since it is hard be determine with certainty that it isn't just
    part of a legit interval of dates used in a sentence.

    "only_snippets" was introduced because the junk_removal is run multiple times: one time to remove non-counted 
    patterns such as dates, before removing duplicates again, and another time to remove the counted junk_snippets after duplicate removal.
    Removing the counted snippets takes a long time, so it is the goal to remove as many rows as possible before removing counted junk.

    Args:
        dataset (Dataset): The dataset to be cleaned
        columns (list): The dataset columns to clean
        junk_snippets (dict): A list with junk snippets
        only_snippets (bool): set to True if the other junk patterns should be ignored, and only junk snippets should be removed. Default to False.

    Returns:
        Dataset: A dataset with junk removed
    """
    import re, hashlib

    # Match numeric date snippets such as: "| Aired: 02/11/2016 ", "| 02/11/2016", "|02/11/2016 - ABC.com"
    # In words: After and including a single "|", match a numeric date, including everything coming before or after it, that is not a "|"
    # Dates can have all possible number separaters (. - /)
    # After analysing the junk data using reg exps, no junk dates were found before a "|", so we only look for dates after "|"
    # There also does not seem to be any (strictly numeric) dates before or after " - "
    date_pattern = r'\|[^\|]*\d{2}[\.\-\/]\d{2}[\.\-\/]\d{4}[^\|]*'

    # catch junk that has had its count split up after "find_junk_in_rows()" because of different dates (written with letters) or times such as:
    # "On This Day in History - July 17th - Almanac - UPI.com|"
    # "TV and radio listings: April 15 - The Washington Post|"
    # "Photos of the Day: July 27"
    # "July 23 (BusinessDesk) -"
    # "Morning trivia quiz: November 16 |"
    # "- Sunday, November 18| Latest News Videos | Fox News"
    # "| 2:19" - Usually after "The Tonight Show Starring Jimmy Fallon" or similar
    # "| Season 21 Episode 131 " - Usually after "Saturday Night Live " or similar.
    # World News Tonight with David Muir: World News 04/20/16:
    # Good Morning America : GMA 09/12/15:
    manually_detected_junk = [r'On This Day in History .* Almanac - UPI\.com\|', 
                              r'TV and radio listings:.*- The Washington Post\|',
                              r'Photos of the Day: .* \d{1,}\s',
                              r'.*\d{1,} \(BusinessDesk\) -',
                              r'.+ trivia quiz:.+\d{1,} \|',
                              r'- .+day, .+ \d{1,}\| Latest News Videos \| Fox News',
                              r'\| .{0,2}:\d{2}',
                              r'\|.*Season \d{1,} Episode \d{1,}',
                              r'World News.*:',
                              r'Good Morning America.*:',
                              r'TV and radio listings: .* \d{1,2} -']
    
    # other manually detected junk such as:
    # \u00a0 \u00a0 \u00a0Presented by Facebook|
    # <strong>, </strong>, <i>, <b>, <u>, <p class="something">, </p>, <br/>, and other html.
    # 
    # The reason all html patterns are written down manually, instead of using a more general pattern, is beacause "<>" is sometimes used in natural language:
    # "Reuters The European Commission has fined Credit Agricole <CAGR.PA>, HSBC <HSBA.L> and JPMorgan Chase <JPM.N> a total of 485 million euros"
    # Something like BeautifulSoup.get_text() also removes <CAGR.PA>, and is not an option either.
    # 
    # Some of the articles that still had html elements, even after the initial cleaning in SciRide, may be bit messed up, even after this second cleaning
    # because they were errornous in nature. They may contain ads such as "SUBSCRIBE TO THIS NEWSPAPER", or multiple articles, since they were not divided correctly initially
    # or may now contain "merged" words like "donebecause", because removing the html moved the innerHTML dicrectly next to each other. These cases are seemingly few, though.
    manually_detected_junk.extend([r'.*Presented by Facebook\|',
                                   r'<\/{0,1}(div|aside|span|table|tbody|thead|tr|td|li|ul|ol|del|br|strong|i|b|u|p|a|h\d)(\s[^<>]*=[^<>]*>|\/{0,1}>)',
                                   r'<(form|section).*>.*</(form|section)>'])

    # remove all snippets from junk_snippets that include dates, since a more general reg exp will be added to catch all dates
    junk_snippets_filtered: list = [snippet for snippet in junk_snippets if re.search(date_pattern, snippet) == None]

    # escape the junk_key characters according to re
    # "|" for instance is an operator in re and needs to be escaped
    junk_snippets_filtered = [re.escape(snippet) for snippet in junk_snippets_filtered]
    
    # a list to hold all junk reg exp patterns
    # each junk_snippet counts as a pattern
    patterns = [date_pattern]
    patterns.extend(manually_detected_junk)
    patterns.extend(junk_snippets_filtered)

    # combine all reg exp patterns to a single pattern, which is just adding "|"(or) between all.
    pattern = re.compile("|".join(patterns)) if not only_snippets else re.compile("|".join(junk_snippets_filtered))

    def remove_junk_from_row(examples):
        """Removes junk from rows according to the created "pattern".

        Args:
            examples (dict): dict with lists of row fields with junk

        Returns:
            dict: dict with lists of row fields without junk
        """
        # init dict of junk_free_rows
        junk_free_rows = {column: [] for column in columns}
        junk_free_rows["new_article_id"] = []

        for i in range(len(examples[columns[0]])):
            merged_sequence = ""
            for column_name in columns:
                # replace everything in the seqeunce, that matches the junk pattern, with ""
                junk_free_sequence = pattern.sub("", examples[column_name][i])

                # remove "fake" words. Descriptions that have been ended abruptly by "..." contains "fake" words:
                # "A radical overhaul of Sydney's public bus transport system could cost commuters up to $1300 in additi..."
                # Remove "additi" ^
                abrupt_ennding_pattern = re.compile(r'\s[a-zA-Z0-9\-\'\"]+\.\.\.$')
                junk_free_sequence = abrupt_ennding_pattern.sub(" ...", junk_free_sequence)

                # replace html characters such as &rsquo; with their unicode equivalent
                junk_free_sequence = replace_html_characters(junk_free_sequence)

                # replace all whitespace with a single space
                space_pattern = re.compile(r'\s{2,}')
                junk_free_sequence = space_pattern.sub(" ", junk_free_sequence)

                # remove all whitespace at the begnining and end of a sentence
                abrupt_ennding_pattern = re.compile(r'^\s*|\s*$')
                junk_free_sequence = abrupt_ennding_pattern.sub("", junk_free_sequence)

                junk_free_rows[column_name].append(junk_free_sequence)

                # add the junk_free seqeunce to the other sequences of the row (usually title and descripion)
                # to be used in creating the new hashed id
                merged_sequence += junk_free_sequence

            junk_free_rows["new_article_id"].append(hashlib.md5(merged_sequence.encode("utf-8")).hexdigest())

        return junk_free_rows
    
    # remove junk from rows in batches
    junk_free_dataset: Dataset = dataset.map(remove_junk_from_row, batched=True, batch_size=10000)

    return junk_free_dataset

def replace_html_characters(text: str):
    """
    Replces html characters with their unicode equivalent:

    A radical overhaul of Sydney&rsquo;s public bus transport system could cost commuters up to $1300 in ...
        |
        v
    A radical overhaul of Sydney’s public bus transport system could cost commuters up to $1300 in ...

    Args:
        text (str): String with HTML characters

    Returns:
        str: String without HTML characters
    """
    return html.unescape(text)


def remove_whole_junk_rows(dataset: Dataset, columns: list = ["title", "description"]):
    """Where remove_junk_from_rows() removes small junk snippest within a row, this function removes the entire row, 
    if it is deemed as total junk. This could be:
    - empty or too short
    - conatains uncleanable html
    

    Args:
        dataset (Dataset): A dataset, which may contain junk rows
        columns (list, optional): List of columns in which to look for junk. Defaults to ["title", "description"].

    Returns:
        Dataset: A dataset with whole junk rows removed
    """
    # caching can be disabled, when the "junk" list needs to be filled up, so the filtered out junk can be analysed.
    # If a cached version of the result is used, junk won't be filled up
    # disable_caching()

    junk = []

    def filter_rows(example):
        """Filtering function. 

        Args:
            example (dict): A dataset row

        Returns:
            bool: Returns true if the example should be kept, False otherwise.
        """
        import re
        columns_text = [example[column] for column in columns]
        merged_text = " ".join(columns_text)
    
        if len(merged_text) <= 16:
            junk.append(merged_text)
            return False

        # a more general pattern '<[^<>]*>' was used to find all articles that contained any possible html element. 
        # The articles were saved in a json file and then the matched articles were manually looked through to find which HTML elements
        # warranted a complete removal of the article and which just had to be cleaned up a bit. 
        # If an article contains "<iframe" it is completely removed, 
        # beccause very few had anything valuable, it was mostly:
        # "{some option} Cast your vote now. All answers are stored anonymously. ADD THIS POLL TO YOUR SITE (copy the code below) <iframe src="http://www.wnd.com/just-plain-nuts-2/" style="width: 600px; height: 582px; border: 1px;"></iframe>"
        # The reason "a href" is separate is because it was spotted without ">"
        # If an article contains HTML that does not fit the below pattern, the html part is simply cleaned (in remove_junk_from_rows()).
        html_patterns = [r'<\/{0,1}(script|iframe)[^<>]*>',
                         r'a href',
                         r'_firefly_ad']
        
        html_pattern = re.compile("|".join(html_patterns))
        if re.search(html_pattern, merged_text) != None:
            junk.append(merged_text)
            return False
        
        return True
    
    filtered_dataset: DatasetDict = dataset.filter(filter_rows)
    print("Rows removed: ", len(junk))
    return filtered_dataset

def remove_non_english_articles():
    #NOTE: This function is not actually used to filter out non-english articles, it was just used as a sanity check, to see if there were any. 
    # There did not seem to be.
    """Detect and remove articles in a "foreign" language (not english).
    """
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0

    def sequence_is_english(seq: str):
        return detect(seq) == "en"
    
    def filter_foreign_rows(row):
        seq = ". ".join([row["title"], row["description"]])
        if (not sequence_is_english(seq)):
            print(seq)
            return False
        return True
    
    dataset = load_from_disk(FILTERED_DATASET_DIR_NAME)
    dataset = dataset["latimes.com"]
    dataset.filter(filter_foreign_rows)

# --- FUNCTIONS FOR LABELING AND MANIPULATING FINE-TUNING DATA ---
def word_tokenize_sequence(seq: str):
    """A word tokenizer used to extract unigrams and bigrams consisting of whole (but stemmed) words.

    Args:
        seq (str): A (possibly) multi-sentence text sequence

    Returns:
        list(str): a list of stemmed unigrams and bigrams (bigrams as 2-token strings separated by space)
    """
    stemmer = PorterStemmer()
    all_stemmed_unigrams = []
    all_stemmed_bigrams = []

    # split (lowered) text into sentences
    sentences = sent_tokenize(seq.lower(), language="english")

    for sentence in sentences:
        # split a sentence into unigrams
        tokens = word_tokenize(sentence, language="english")

        # stem every token that is not punctuation and save in a list
        stops = list(string.punctuation) + ["'s", "``", "’", "''", "--", "—", "–", "“", "”"]
        stemmed_unigrams = ([stemmer.stem(token)
                        for token in tokens if token not in stops])
        all_stemmed_unigrams.extend(stemmed_unigrams)

        # create a list of bigrams (joined in a string by a space) from all the stemmed unigrams
        # bigrams are created per sentence, rather than from all unigrams, to avoid creating bigrams with tokens from different sentences.
        stemmed_bigrams_as_strings = [" ".join([bigram[0], bigram[1]]) for bigram in bigrams(stemmed_unigrams)]
        all_stemmed_bigrams.extend(stemmed_bigrams_as_strings)

        # add the bigrams to the list of unigrams
        stemmed_ngrams = all_stemmed_unigrams + all_stemmed_bigrams

    return stemmed_ngrams

def create_seq_class_dataset_mc_ml(class_keywords: dict, dataset: DatasetDict, labeled_dataset_size: int = 0.01):
    """Create dataset for multi-class, multi_label sequence classification training by extracting them from an exisiting dataset,
    and labeling the rows using the provided classes and keywords. By multi-class, it is meant that there are more than two options
    for a classification, in other words it could be "cat/dog/fish/cow" and not "is_nagative/is_positive". By multi-label, it is meant,
    that each sequence can have more than one label, in other words "Donald Trump calls Joe Biden 'Sleepy Joe'", has two labels: 
    "donald trump" and "joe biden".

    Args:
        class_keywords (dict): A dictionary with they keys being classes and values being associated (stemmed and lowered) keywords
        dataset (DatasetDict): A Dataset Dict, typically split up into Datasets for each outlet
        labeled_dataset_size (float): How big should the labeled dataset be (max), as a fraction of the full dataset. For instance, 0.01 for 1% of the full dataset.
    
    Returns: The labeled dataset with an (attempted) even distribution of classes    
    """
    from datasets import concatenate_datasets, ClassLabel, Sequence

    classes = list(class_keywords.keys())

    # dict for mapping a class name to a label id
    label_map_cls2id = {classes[i]: i for i in range(len(classes))}

    # dict for holding the counts of each label in the labeled dataset
    label_counts = {label_map_cls2id[cls]: 0 for cls in classes}

    # shuffle the dataset to ensure the samples are randomly distributed with regards to outlet and date
    shuffled_dataset = dataset.shuffle(seed=2022)

    # calculating the maximum size, that the smallest class_dataset can be, in order to not surpass the goal size of the full labeled dataset,
    # once all class_datasets have been decreased to match the size of the smallest class_dataset.
    # Essentially, if all classes have equally large datasets, how big can they be, such that when combined, they are <= to some fraction of the full dataset
    max_size_of_min_class = (shuffled_dataset.num_rows * labeled_dataset_size) / len(classes)
    
    def is_of_class(tok_seq: list, cls: str):
        """Checks whether a text sequence belongs to a certain class by looking at the keywords associated with cls

        Args:
            tok_seq (list[str]): a list of strings (unigrams and bigrams) extracted from a single coherent document
            cls (str): a topic class, used to lookup which associacted keywords to search for in the tok_seq

        Returns:
            boolean: True if tok_seq contains keywords associated with cls, False otherwise
        """
        for keyword in class_keywords[cls]["keywords_stemmed"]:
            if keyword in tok_seq:
                return True
        return False

    def add_labels(examples):
        """Meant to be used as input to the Dataset.map() function. For every row in the dataset, it is tested whether the sentence
        belongs to any of the classes provided by class_keywords. If a class match is found, the corrensponding label is added to the list of 
        labels associated with each sequence. If a row does not match any class, it is discarded, meaning it is not returned with the rest of the batch.
        Meant to be used with batched mapping.

        Args:
            examples (dict): The Dataset rows

        Returns:
            dict: A labeled row.
        """
        print(label_counts)
        labeled_rows = {"title": [], "description": [], "meta": [], "labels": [], "new_article_id": []}

        # skip labeling, if the maximum amount of labeled rows have been reached
        if (min(label_counts.values()) >= max_size_of_min_class):
            return labeled_rows
        
        for i in range(len(examples["title"])):
            title, description, meta, new_article_id = examples["title"][i], examples["description"][i], examples["meta"][i], examples["new_article_id"][i]

            # initializing list of labels for the current sequence
            seq_labels = []

            # combining title and description into single article sequence
            merged_text = ". ".join([title, description])

            # tokenize sequence into list of stemmed unigram and bigrams
            tok_seq = word_tokenize_sequence(merged_text)

            # determine the label(s) of the sequence
            for cls in classes:
                label = label_map_cls2id[cls]
                if is_of_class(tok_seq, cls):
                    seq_labels.append(label)

            if seq_labels != []:
                labeled_rows["title"].append(title)
                labeled_rows["description"].append(description)
                labeled_rows["meta"].append(meta)
                labeled_rows["new_article_id"].append(new_article_id)
                labeled_rows["labels"].append(seq_labels)

            for label_id in seq_labels:
                label_counts[label_id] += 1

        return labeled_rows

    # adding the correct labels to the rows in the dataset
    labeled_dataset = shuffled_dataset.map(add_labels, batched=True, batch_size=10000)

    # making sure there are an equal amount of each label in the dataset (detmined by the smallest class)
    class_datasets = [labeled_dataset.filter(lambda row: label_map_cls2id[cls] in row["labels"]) for cls in class_keywords]
    dataset_sizes = [dataset.num_rows for dataset in class_datasets]
    min_dataset_size = min(dataset_sizes)
    even_class_datasets = [dataset.select(range(min_dataset_size)) for dataset in class_datasets]
    merged_labelled_dataset: Dataset = concatenate_datasets(even_class_datasets)

    # Since each sequence can have multiple labels, the same sequence can appear in multiple class_datasets 
    # resulting in duplicates when concatinating. We can remove the duplicates (only keeping 1), to try to keep the distribution of labels even.

    # Example: If the Trump dataset has 10 rows and the Biden dataset has 10 rows, but exactly one of the rows in each dataset is identical to the other,
    # due to the labels overlapping, that means, that when combining the datasets, the Trump and Biden datasets will get an 11th row each, because they both gave the other
    # an extra row. By removing one of these duplicates, resulting in 19 rows combined, there will be 10 Trump rows and 10 Biden rows,
    # they will just be sharing one.

    # Note that this approach of removing duplicates does not work that well to keep the distribution of labels even.
    # There are cases where two classes share no articles but one of them still have artilces, which include the other class's label. 
    # In that case one of those classes occur more than the other, which is why create_seq_class_dataset_mc_ml_reselect() ended up being necessary.

    # remove_duplicates
    merged_labelled_dataset = remove_duplicates(merged_labelled_dataset, id_column="new_article_id")

    # making sure the labels column is of the Sequence(ClassLabel) type
    # this needs to be done after removing duplicates as it seems the column type is not maintained in the pandas df
    merged_labelled_dataset = merged_labelled_dataset.cast_column("labels", Sequence(ClassLabel(num_classes=len(classes), names=classes)))

    return merged_labelled_dataset

def check_multi_class_count(dataset: DatasetDict):
    """Checks how many of each class occurs in the labeled multi-label data.
    """
    counts = {}

    def count (example):
        for label in example["labels"]:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

    dataset.map(count)
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

def check_multi_multi_count(dataset: DatasetDict):
    """Checks how many articles have multiple labels
    """
    counts = {}

    def count (example):
        label_count = len(example["labels"])
        if label_count in counts:
            counts[label_count] += 1
        else:
            counts[label_count] = 1

    dataset.map(count)
    return counts

def create_seq_class_dataset_mc_ml_reselect(class_keywords: dict, labeled_dataset: Dataset, labeled_dataset_size: int = 0.01, exclude_article_ids={}):
    """This is meant to be a "relabeling" or a "selection phase" of an already labeled multi-label dataset.
    In the case of this project, the "labeled_dataset" comes from a badly implemented .create_seq_class_dataset_mc_ml() which did not properly distribute classes. 
    Ideally, labeled_dataset should come from a function that labeled the data without even attempting to make an even distribution, because the frequency
    of the classes is needed for the function to find the least to most common classes.

    Args:
        class_keywords (dict): A dictionary with they keys being classes and values being associated (stemmed and lowered) keywords
        dataset (DatasetDict): A labeled multi-label dataset
        labeled_dataset_size (float): How big should the new labeled dataset be (max), as a fraction of the full labeled dataset. For instance, 0.90 for 90% of the full dataset.
        added_article_ids (dict): dict with null values and article ids as keys, used to specify which articles should not be included if they are found
    
    Returns: A relabeled dataset with an even distribution of classes   
    """
    from datasets import concatenate_datasets

    # disable to ensure counting (during .map()) is not skipped
    disable_caching()

    classes = list(class_keywords.keys())

    # dict for mapping a class name to a label id
    label_map_cls2id = {classes[i]: i for i in range(len(classes))}

    # dict for holding the counts of each label in the labeled dataset
    # used to keep track of each label count to not exceed the goal
    final_label_counts = {label_map_cls2id[cls]: 0 for cls in classes}

    # count the occurences of each class to find out which class is least common in the dataset
    # used to decide which class to look for first.
    # The intuition is that the least common class should have the least resctrictions when being selected.
    # Example: if all other classes have already reached the target size, and it's time for the last class to be selected, it will have the added restriction
    # that it can only inlucde singly labeled samples, because if samples with multiple labels are selected, some of the other 
    # classes will exceed the goal. The most common class should have less of a problem with those restrictions.
    initial_label_counts = check_multi_class_count(labeled_dataset)

    # sort the counts increasing, because the smallest should be used first
    sorted_initial_label_counts = dict(sorted(initial_label_counts.items(), key=lambda item: item[1]))

    # dict to keep track of the already added articles
    added_article_ids = exclude_article_ids

    # shuffle the dataset to ensure the samples are randomly distributed with regards to outlet and date
    shuffled_dataset = labeled_dataset.shuffle(seed=2022)

    # calculating the maximum size, that the smallest class_dataset can be, in order to not surpass the goal size of the full labeled dataset,
    # once all class_datasets have been decreased to match the size of the smallest class_dataset.
    # Essentially, if all classes have equally large datasets, how big can they be, such that when combined, they are <= to some fraction of the full dataset
    max_size_of_min_class = (shuffled_dataset.num_rows * labeled_dataset_size) / len(classes)

    def add_labels(examples, current_label_id):
        """Meant to be used as input to the Dataset.map() function. For each row, check if labels include current_label_id. If it does, it is returned and if it does not,
        it is discarded. If it does but it or any of the other labels have already reached the target size, it is also discarded.
        Meant to be used with batched mapping.

        Args:
            examples (dict): The Dataset rows
            current_label_id (int): The current id to look for. Rows are only added if they belong to this label.

        Returns:
            dict: A labeled row.
        """
        print(final_label_counts)
        labeled_rows = {"title": [], "description": [], "meta": [], "labels": [], "new_article_id": []}

        # skip labeling, if the maximum amount of labeled rows have been reached
        if (min(final_label_counts.values()) >= max_size_of_min_class or final_label_counts[current_label_id] >= max_size_of_min_class):
            return labeled_rows
        
        for i in range(len(examples["title"])):
            title, description, meta, new_article_id, labels = examples["title"][i], examples["description"][i], examples["meta"][i], examples["new_article_id"][i], examples["labels"][i]

            # avoid duplicates by skipping the iteration when the article has already been added before
            # this is necessary because the .map(add_labels) function is run multiple times, one time for each class
            if new_article_id in added_article_ids:
                continue

            skip_row = False
            for label in labels:
                # if just one of the classes have already exceeded the goal, the loop breaks because the article should not be included
                if final_label_counts[label] >= max_size_of_min_class:
                    skip_row = True
                    break

            if skip_row:
                continue

            # if the current label we are looking for is not in the labels, nothing is added to the rows
            if current_label_id in labels:
                labeled_rows["title"].append(title)
                labeled_rows["description"].append(description)
                labeled_rows["meta"].append(meta)
                labeled_rows["new_article_id"].append(new_article_id)
                labeled_rows["labels"].append(labels)
            else:
                # if nothing is added, counts and added_article_ids do not need to be updated, so just skip the rest
                continue

            for label_id in labels:
                final_label_counts[label_id] += 1
            
            added_article_ids[new_article_id] = None

        return labeled_rows

    # adding the correct labels to the rows in the dataset 
    # starting with the least common class
    labeled_datasets = [shuffled_dataset.map(lambda examples: add_labels(examples, label_id), batched=True, batch_size=1000) for label_id in sorted_initial_label_counts]
    del added_article_ids

    merged_labelled_dataset: Dataset = concatenate_datasets(labeled_datasets)
    del labeled_datasets

    return merged_labelled_dataset

def create_seq_class_dataset_covid(class_keywords: dict, dataset: DatasetDict, labeled_dataset_size: int = 0.01):
    """Create dataset for binary single-label sequence classification with the label being either "is_covid" or "is_not_covid". 
    Very similar to create_seq_class_dataset_mc_ml[_reselect], but much simpler implementation, because there are no label-overlaps caused by the multi-label structure.

    Args:
        class_keywords (dict): A dictionary with they keys being classes and values being associated (stemmed and lowered) keywords
        dataset (DatasetDict): A Dataset Dict, typically split up into Datasets for each outlet
        labeled_dataset_size (float): How big should the labeled dataset be (max), as a fraction of the full dataset. For instance, 0.01 for 1% of the full dataset. (Note: +- a batch size, and rounded to the next whole number)
    
    Returns: A labeled dataset
    """
    from datasets import concatenate_datasets, ClassLabel

    classes = list(class_keywords.keys())

    # dict for mapping a class name to a label id
    label_map_cls2id = {classes[i]: i for i in range(len(classes))}

    # dict for holding the counts of each label in the labeled dataset
    label_counts = {label_map_cls2id[cls]: 0 for cls in classes}

    # shuffle the dataset to ensure the samples are randomly distributed with regards to outlet and date
    shuffled_dataset = dataset.shuffle(seed=2022)

    # calculating the maximum size, that the smallest class_dataset can be, in order to not surpass the goal size of the full labeled dataset,
    # once all class_datasets have been decreased to match the size of the smallest class_dataset.
    # Essentially, if all classes have equally large datasets, how big can they be, such that when combined, they are <= to some fraction of the full dataset
    max_size_of_min_class = (shuffled_dataset.num_rows * labeled_dataset_size) / len(classes)

    def add_labels(examples):
        """Meant to be used as input to the Dataset.map() function. For every row in the dataset, it is tested whether the sentence
        belongs to any of the classes provided by class_keywords. If a class match is found, the corrensponding label is added to the row.
        If no match is found, the row is discarded, meaning it is not returned with the rest of the batch.
        Meant to be used with batched mapping.

        Args:
            examples (dict): The Dataset rows

        Returns:
            dict: a dict with lists of columns, one of them a "label" column.
        """
        print(label_counts)
        # dict for holding the rows that get a label
        labeled_rows = {"title": [], "description": [], "meta": [], "label": [], "new_article_id": []}

        # skip labeling, if the maximum amount of labeled rows have been reached
        if (min(label_counts.values()) >= max_size_of_min_class):
            return labeled_rows

        # loop through the batch
        for i in range(len(examples["title"])):
            title, description, meta, new_article_id = examples["title"][i], examples["description"][i], examples["meta"][i], examples["new_article_id"][i]

            # initializing label for the current sequence
            seq_label = -1

            # combining title and description into single article sequence
            merged_text = ". ".join([title, description])

            # tokenize sequence in list of stemmed unigram and bigrams
            tok_seq = word_tokenize_sequence(merged_text)

            # determine the label of the sequence
            # sentences from before the covid outbreak are labeled "is_not_covid"
            if (meta["date"] < "20200101"):
                seq_label = label_map_cls2id["is_not_covid"]

            # sentences containing the correct keywords are labeled "is_covid"
            for keyword in class_keywords["is_covid"]["keywords_stemmed"]:
                if keyword in tok_seq:
                    seq_label = label_map_cls2id["is_covid"]

            # add the row to labeled_rows if a label fit its sequence
            if (seq_label != -1):
                labeled_rows["title"].append(title)
                labeled_rows["description"].append(description)
                labeled_rows["meta"].append(meta)
                labeled_rows["new_article_id"].append(new_article_id)
                labeled_rows["label"].append(seq_label)

            # update the number of occurences of each class
            if seq_label != -1: label_counts[seq_label] += 1

        return labeled_rows

    # adding the correct labels to the rows in the dataset, disicarding a row if it has not label
    labeled_dataset = shuffled_dataset.map(add_labels, batched=True, batch_size=10000)

    # making sure there are an equal amount of each label in the dataset (determined by the smallest class)
    class_datasets = [labeled_dataset.filter(lambda row: label_map_cls2id[cls] == row["label"]) for cls in class_keywords]
    dataset_sizes = [dataset.num_rows for dataset in class_datasets]
    min_dataset_size = min(dataset_sizes)
    even_class_datasets = [dataset.select(range(min_dataset_size)) for dataset in class_datasets]
    merged_labelled_dataset: Dataset = concatenate_datasets(even_class_datasets)

    # making sure the label column is of the ClassLabel type
    merged_labelled_dataset = merged_labelled_dataset.cast_column("label", ClassLabel(num_classes=len(classes), names=classes))

    return merged_labelled_dataset
    
def remove_labeled_subset_from_full_data(full_data: Dataset):
    """Remove a row from the pre-training dataset, if it is identical to a labeled row, 
    so there is no overlap between fine-tuning data and pre-training data.
    """
    labeled_covid = merge_split_dataset(load_from_disk(LABELED_COVID_CLASS_DATASET_DIR_NAME))
    labeled_multi = merge_split_dataset(load_from_disk(LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME))

    # use dict to make use of hashing, making the "in" function much faster.
    labeled_ids = {meta["article_id"]: None for meta in labeled_covid[:]["meta"] + labeled_multi[:]["meta"]}

    partial_data = full_data.filter(lambda example: example["meta"]["article_id"] not in labeled_ids)

    partial_data.save_to_disk(PARTIAL_DATASET_DIR_NAME)

def interactive_NER_labeler():
    """An interactive script, which presents the user with setences from the LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME,
    and asks for each word, if it's a named entity. Saves the resulting words and labels in a JSON-file.
    """

    # define the colors to highlight words
    class bcolors:
        GREEN = '\033[92m'
        ENDC = '\033[0m'

    import sys

    # Create or open a JSON-file to save the results in
    try:
        with open("%s.json" % LABELED_NER_JSON_FILE_NAME) as json_file:
            labeled_ner_dict = json.load(json_file)
    except:
        labeled_ner_dict = {"rows": []}
    
    # open the NER labelmap so the user knows what the label_ids mean
    with open("blobs/NER_labelmap.json") as json_file:
            label_map = json.load(json_file)
    
    # add a "skip" option to the labelmap
    label_map["O remaining words"] = 99

    def prompt_label(row):
        """A function to be run for each row. It splits a row up into sentences and prompts the user for each word in each every sentence.

        Args:
            row (dict): A dict of row fields.
        """
        # create list of sentences from title and descriptions
        sentences = sent_tokenize(row["title"]) + sent_tokenize(row["description"])

        for sentence in sentences:
            words = sentence.split(" ")
            labels = []
            for i, word in enumerate(words):
                print(label_map)

                # highlight the current word
                colored_list = words.copy()
                colored_list[i] = bcolors.GREEN + word + bcolors.ENDC
                print(" ".join(colored_list))

                # ask the user for input on the word
                str_label = input(word + ": ")

                # if input is empty, the label is 0, otherwise cast input to in
                label = 0 if str_label == '' else int(str_label)

                # if input is associated with a label, add the label to labels otherwise "skip" the rest of words by labeling them 0
                if label <= 6:
                    labels.append(label)
                else:
                    rest = [0 for j in range(i, len(words))]
                    labels.extend(rest)
                    break
            
            labeled_row = {"words": words, "labels": labels, "meta": row["meta"]}
            labeled_ner_dict["rows"].append(labeled_row)

    # get sys args which defines the size and interval of the batch to be labeled
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])

    reference_dataset_name = LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME
    dataset: DatasetDict = load_from_disk(reference_dataset_name)

    # select from start (inluding) to end (including)
    sub_dataset: Dataset = dataset["train"].select(range(start_index, end_index+1))
    sub_dataset.map(prompt_label)

    # save the results
    with open(LABELED_NER_JSON_FILE_NAME+".json", "w+", encoding='utf-8') as f:
        json_record = json.dumps(labeled_ner_dict)
        f.write(json_record)

def save_NER_json_as_dataset():
    """Go thorugh the NER-labeled json-file of articles and restructure it to be saved as a Hugging Face dataset.
    """
    from datasets import ClassLabel, Sequence
    labeled_ner_dict = load_json("blobs/labeled_ner_json.json")

    dataset_dict = {}
    for key in labeled_ner_dict["rows"][0]:
        dataset_dict[key] = []

    for dict in labeled_ner_dict["rows"]:
        for key, value in dict.items():
            dataset_dict[key].append(value)

    ner_dataset: Dataset = Dataset.from_dict(dataset_dict)

    labels = [key for key in load_json("blobs/NER_labelmap.json")]

    label_class = Sequence(ClassLabel(num_classes=len(labels), names=labels))
    ner_dataset = ner_dataset.cast_column("labels", label_class)

    save_dataset_as_arrow(ner_dataset, LABELED_NER_DATASET_DIR_NAME, True, True)

def create_and_save_split_for_NER(dataset: DatasetDict, wanted_names: list = ["vladimir", "putin", "angela", "merkel"]):
    """Create a train-test split from the NER data such that 2 of the chosen poltician names from "multi_class_keywords.json" 
    appear only in test and not in train. In this case: Vladimir Putin and Angela Merkel

    Args:
        dataset (DatasetDict): Labeled NER dataset with names of politicians
        wanted_names: A list of wanted names, which should only occur in the test split
    """
    def extract_specific_names(example):
        """Filtering function.

        Args:
            example (dict): A row of fields

        Returns:
            bool: True if the example contains any of the wanted names, False otherwise.
        """
        if any(name in [word.lower() for word in example["words"]] for name in wanted_names):
            return True
        
        return False
    
    split_dataset = DatasetDict()
    # add the articles containing wanted_names to the test split
    split_dataset["test"] = merge_split_dataset(dataset.filter(extract_specific_names))

    # use dict to make use of hashing, making the "in" function much faster.
    test_words = {" ".join(words): None for words in split_dataset["test"][:]["words"]}

    # removing the extracted test data from the full data to make the train data
    split_dataset["train"] = merge_split_dataset(dataset.filter(lambda example: " ".join(example["words"]) not in test_words))

    # create empty validation dataset as it is not necessary and training data is sparse, but the list, empty or not, need to exist during training.
    split_dataset["validation"] = Dataset.from_dict({key: [] for key in split_dataset["train"].features})

    save_dataset_as_arrow(split_dataset, LABELED_NER_DATASET_NAMESPLIT_DIR_NAME, False, False)

def generalize_NER_dataset(dataset: Dataset):
    """Transform an NER dataset with different label subtypes such as PER and ORG into a dataset where the labels are more generel, so: not NE(O), B-NE or I-NE.

    Args:
        dataset (Dataset): The dataset with labels ranging from 0 to beyond 2

    Returns:
        Dataset: The transformed dataset
    """
    from datasets import ClassLabel, Sequence
    def generalize(row):
        # even numbers become 2, uneven become 1, 0 stays 0
        gen_labels = []
        for label in row["labels"]:
            if label == 0:
                gen_labels.append(0)
            elif label%2 == 1:
                gen_labels.append(1)
            else:
                gen_labels.append(2)
        return {"labels": gen_labels}
    
    dataset: Dataset = dataset.map(generalize)

    label_class = Sequence(ClassLabel(num_classes=3, names=["O", "B-NE", "I-NE"]))
    dataset.cast_column("labels", label_class)

    return dataset
    
def substitute_keywords_in_rows(dataset: DatasetDict, keywords: list, substitute_string: str, columns: list = ["title", "description"]):
    """Given a list of keywords, go through a dataset and substitute each occuring keyword with substitute_string

    Args:
        dataset (DatasetDict): The Dataset in which to substitute keywords
        keywords (list): The keywords to be substituted
        substitute_string (str): The string with which the keywords should be replaced
        columns (list, optional): The columns in which to look for the keywords. Defaults to ["title", "description"].

    Returns:
        DatasetDict: Dataset with keywords substituted with substitute_string
    """
    import re
    pattern = re.compile("|".join(keywords), flags=re.IGNORECASE)

    def substitute_keywords(row):
        amended_row = {column: "" for column in columns}
        for column in columns:
            amended_row[column] = pattern.sub(repl=substitute_string, string=row[column])
        return amended_row
    
    dataset = dataset.map(substitute_keywords)
    return dataset

def pack_inputs_like_roberta(dataset: Dataset, tokenizer: PreTrainedTokenizerFast):
    """
    Roberta paper:
    "Full-Sentence: Each input is packed with full sentences sampled contiguously from one
    or more documents, such that the total length is at most 512 tokens.

    Doc-Sentence: Inputs are constructed similarly to FULL-SENTENCES, except that they
    may not cross document boundaries."

    Roberta uses full-sentence, but they also train on large documents, so most inputs will be packed with sentences from the same document, 
    and occasionally, there will be an overlap between documents.
    We train on smaller documents (typically < 512), so using full-sentence, there would often be overlaps between documents, possibly multiple documents in one input.
    Therefore, we use doc-sentence, which also performs slightly better according to roberta paper, and intuitively it makes the most sense, since it only places sentences,
    that were originally written together, next to each other.

    Args:
        dataset (Dataset): A dataset of titles and descriptions.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to tokenize the articles in order to count tokens.

    Returns:
        Dataset: A dataset of articles with full sentences (title + description). 
                 If title + description exceeds 512 characters, sentences from 512 and up will be moved to separate row.
    """
    import hashlib

    def pack_input(examples):
        """Go through a batch of articles (title+description) and repack it such that no input will contain more than 512 tokens after tokenization.
        The articles a tokenized during counting and then un-tokenized before saving, where title and description is saved as a single article.

        Args:
            examples (dict): Dict of lists of row fields.

        Returns:
            dict: Dict of lists of row fields where the 
        """
        repackaged_inputs = {"article": [], "meta": [], "new_article_id": []}

        for i in range(len(examples["title"])):
            # list to hold articles cosisting of sentences < 512 tokens
            max_512_articles = []
            title, description, meta = examples["title"][i], examples["description"][i], examples["meta"][i]

            sentences = []
            sentences.extend(sent_tokenize(title, language="english"))
            sentences.extend(sent_tokenize(description, language="english"))

            # creating a dict from the items in the list, will remove duplicates due to the nature of dicts
            sentences = list(dict.fromkeys(sentences))

            # list to hold all tokens from the article's sentences, until it reaches max
            article = []
            for i in range(len(sentences)):
                tokens = tokenizer.tokenize(sentences[i], add_special_tokens=False)

                # add full stop character if there is none. Could often be the case with titles. 
                # There needs to be one, because the sentence will be combined with another,
                # so a full stop needs to be indicated.
                if tokens[-1] not in [".", "?", "!","Ġ...", "..."]:
                    tokens.append(".")
                
                # add space after sentence no matter what, so there is space between sentences when they are combined
                tokens.append(" ")

                # add the tokens to the article if there is room
                if len(article) + len(tokens) < 512:
                    article.extend(tokens)
                # otherwise add the full article to the list of articles and reset the article to the tokens of the sentences there was no room for
                else: 
                    # remove last element in the article's token list, because it is a space character
                    article = article[:-1]
                    max_512_articles.append(tokenizer.convert_tokens_to_string(article))
                    article = tokens
               
            # After looking at all sentences in the article, add the last bit of the article to the list of 512-token articles
            if article != []:
                article = article[:-1]
                max_512_articles.append(tokenizer.convert_tokens_to_string(article))

            # add article pieces to the batch
            repackaged_inputs["article"].extend(max_512_articles)

            # add a meta field for each article piece
            repackaged_inputs["meta"].extend([meta] * len(max_512_articles))

        # create hashed ids for all article pieces
        ids = [hashlib.md5(article.encode("utf-8")).hexdigest() for article in repackaged_inputs["article"]]
        repackaged_inputs["new_article_id"] = ids
        return repackaged_inputs

    dataset = dataset.map(pack_input, batched=True, remove_columns=["title", "description"])

    return dataset

def label_dataset_for_title_description_match(dataset: Dataset):
    """Given a dataset of matching titles and descriptions, scrambles half of the descriptions, so they don't match the titles. Adds labels according to whether the row
    has matching columns or not. 1 for matching 0 for not. 
    To be used in classification fine-tuning where the model should predict, whether the description matches the title.

    Args:
        dataset (Dataset): Dataset with mathcing titles and descriptions

    Returns:
        dataset: Dataset with half matching titles and descriptions, half not.
    """
    from datasets import concatenate_datasets, ClassLabel
    def derange_descriptions(examples):
        """Given a batch of rows, derange the title and description, such that no title is matched with its original description.

        Args:
            examples (dict): Dict of lists of row fields.

        Returns:
            Dataset: Labeled dataset with deranged titles and descriptions.
        """
        # if the last batch only has 1 element it cannot be deranged, so just discard it
        if len(examples["title"]) == 1:
            return {"title": [], "description": []}

        # switch first half of descriptions with second half
        half = int(len(examples["description"])/2)
        return {"description":  examples["description"][half:] + examples["description"][:half] }

    whole = len(dataset)
    half = int(whole/2)

    # split the dataset in to halves
    true_label_dataset = dataset.select(range(0, half))
    false_label_dataset = dataset.select(range(half, whole))

    # shuffle descriptions on second half
    false_label_dataset = false_label_dataset.map(derange_descriptions, batched=True, batch_size=4)

    # add new labels
    true_label_dataset = true_label_dataset.add_column("label", [1] * half)
    false_label_dataset = false_label_dataset.add_column("label", [0] * len(false_label_dataset))

    # merge two halves
    labeled_dataset = concatenate_datasets([true_label_dataset, false_label_dataset])

    # cast the label column to ClassLabel with named classes
    labeled_dataset = labeled_dataset.cast_column("label", ClassLabel(num_classes=2, names=["no_match", "match"]))

    return labeled_dataset


# --- FUNCTIONS FOR VARIOUS UTILITY ---
def save_dataset_as_arrow(dataset: Dataset, dir: str, shuffle=False, split=False, split_size: int = 0.9):
    """Saves a Dataset to disk with the option of shuffling or splitting it.

    Args:
        dataset (Dataset): The dataset to save
        dir (str): Where and under which name to save the dataset.
        shuffle (bool, optional): Shuffles the dataset if True. Defaults to False.
        split (bool, optional): Create train-test-val split of True. Defaults to False.
        split_size (int, optional): The size of train split relative to test, and test relative to val. Defaults to 0.9.
    """
    dataset = dataset.shuffle(seed=2022) if shuffle else dataset
    dataset = create_split_from_scratch(dataset, train_size=split_size) if split else dataset

    dataset.save_to_disk(dir)

def create_split_from_scratch(dataset: Dataset, shuffle=True, stratify_by_column: str = None, train_size: int = 0.90):
    """Splits a dataset up into DatasetDict with train, test and validation keys. This custom function is created beacuse train_test_split does not
    create validation split.

    Args:
        dataset (Dataset): _description_
        shuffle (bool, optional): whether to shuffle the data before splitting. Defaults to True.
        stratify_by_column (str, optional): Whether to stratify by column (maintainging the distribution of values in the specified column). Defaults to None.
        train_size (int, optional): The size of train split relative to test, and test relative to val. Defaults to 0.9.

    Returns:
        DatasetDict: A dataset split up into train, test and validation.
    """
    # splitting the train portion (which is currently 100% of the data) into a train portion (90%) and a test portion (10%)
    train_test_dataset = dataset.train_test_split(train_size=train_size, shuffle=shuffle, seed=2022, stratify_by_column=stratify_by_column)
    
    # splitting the test portion into a test-validation split with the same proportions as the train-test split, the test-split being the largest
    test_val_dataset = train_test_dataset["test"].train_test_split(train_size=train_size, shuffle=shuffle, stratify_by_column=stratify_by_column)

    # assigning the two portions to a test portion and a validation portion
    train_test_dataset["test"] = test_val_dataset["train"]
    train_test_dataset["validation"] = test_val_dataset["test"]

    return train_test_dataset

def split_dataset(dataset: Dataset, n: int = 10):
    """Split dataset into n equally big splits. Used to distrubute the load when removing duplicates.
    Ended up not being necessary.

    Args:
        dataset (Dataset): The dataset to split
        n (int, optional): The amount of datasets to split into. Defaults to 10.

    Returns:
        DatasetDict: A DatasetDict containing n Datasets of size len(dataset) / n
    """
    dataset_dict = DatasetDict()

    split_size = int(dataset.num_rows/n)

    # adding the first splits
    for i in range(0,n-1):
        dataset_dict[str(i)] = dataset.select(range(split_size*i, split_size*(i+1)))

    # adding the last split
    dataset_dict[str(n-1)] = dataset.select(range(split_size*(n-1), dataset.num_rows))

    return dataset_dict

def merge_split_dataset(split_dataset: DatasetDict, shuffle=False):
    """Combines a split dataset into a single dataset.

    Args:
        split_dataset (DatasetDict): The split dataset.
        shuffle (bool, optional): Whether to shuffle the dataset after merging. Defaults to False.

    Returns:
        Dataset: The merged dataset
    """

    from datasets import concatenate_datasets

    merged_dataset = concatenate_datasets(list(split_dataset.values()))
    merged_dataset = merged_dataset.shuffle(seed=2022) if shuffle else merged_dataset

    return merged_dataset

def remove_duplicates(dataset: Dataset, id_column: str = None):
    """More memory efficient approach to removing duplicates. Because article_ids are hashed from title+description, they can be used to confirm the uniqueness of rows.
    By only loading ids into a dataframe when using .drop_duplicates() memory can be saved.

    Args:
        dataset (Dataset): A dataset from which duplicates should be remove. Need to have a meta column containing objecs with an article_id, unless id_column is specified.
        id_column (str): name of the column to match duplicate on. If None, extract ids from meta column.

    Returns:
        Dataset: dataset without duplicates
    """
    from pandas import DataFrame
    def add_ids(rows):
        """Moves the article_ids from the meta column to its own column

        Args:
            rows (dict): Dict of lists of row fields.

        Returns:
            dict: Dict containg new id column
        """
        ids = [meta["article_id"] for meta in rows["meta"]]
        return {"id": ids}

    # save column names before mapping to remember what to remove later
    column_names = dataset.column_names

    print("Adding id column")
    # add a separate id column with the article_ids from meta columns
    dataset_with_ids = dataset.map(add_ids, batched=True, batch_size=10000) if id_column == None else dataset

    # remove everthing except the id column
    dataset_only_ids = dataset_with_ids.remove_columns([name for name in column_names if name != id_column])

    # add ids to dataframe
    print("Loading dataframe")
    pandas_df = DataFrame(dataset_only_ids)

    print("Adding index column")
    # add index column from 0 to num_rows 
    pandas_df["index"] = list(range(dataset_with_ids.num_rows))
    # make sure pandas uses 32bit ints to save memory
    pandas_df.astype({"index": 'int32'})

    print("Dropping duplicates")
    subset_column = "id" if id_column == None else id_column
    pandas_df: DataFrame = pandas_df.drop_duplicates(subset=[subset_column])

    # move the remaining indices to a list
    indices = pandas_df["index"].values.tolist()
    del pandas_df

    # select only the rows associated with an index that was not associated with a duplicate id
    nodup_dataset: Dataset = dataset.select(indices)
    del indices

    print("duplicates removed:", dataset.num_rows - nodup_dataset.num_rows, "/", dataset.num_rows)

    return nodup_dataset

def load_json(file_path):
    with open(file_path) as json_file:
        as_dict = json.load(json_file)
    return as_dict

def save_json(dict, file_path, mode="w+", encoding="utf-8"):
    with open(file=file_path, mode=mode, encoding=encoding) as f:
        json_record = json.dumps(dict)
        f.write(json_record)

# --- FUNCTIONS TO WRAP AROUND IMPL FUNCTIONS TO NEATLY CONTAIN THEM ---
def create_and_save_dataset_of_unique_articles():
    """A wrapper function to create and save full arrow dataset without duplicate articles. 
    It is just calls to various functions, that load the data from json files, remove most duplicates and then eventually removes all duplicates.
    The dataset does not benefit from being shuffled at this point.
    Saves the cleaned dataset as HuggingFace dataset.
    """
    # remove duplicates within each outlet
    dataset_full: DatasetDict = load_dataset_from_files()
    dataset_no_dups_splits = remove_duplicates_within_splits(dataset_full)

    # merga the dataset, that was split up into outlets, and remove remaning duplicates
    dataset_no_dups = remove_duplicates(merge_split_dataset(dataset_no_dups_splits))
    save_dataset_as_arrow(dataset_no_dups, DATASET_DIR_NAME, shuffle=False, split=False)

def remove_dates_and_save(article_dataset):
    """Calls remove_junk_from_rows() with the junk_snippets parameter empty to only remove "universal junk" such as article dates an html elements.
    Saves the cleaned dataset as HuggingFace dataset.
    """
    cleaned_dataset = remove_junk_from_rows(article_dataset, columns=["title", "description"], junk_snippets={}, only_snippets=False)
    save_dataset_as_arrow(cleaned_dataset, DATASET_NODATES_DIR_NAME, shuffle=False, split=False)

def remove_duplicates_and_save(dataset: Dataset, save_dir_name: str, id_column):
    """Simply call remove_dulpicates() and saves as Hugging Face Dataset.
    """
    dataset_no_dups = remove_duplicates(dataset, id_column=id_column)
    save_dataset_as_arrow(dataset_no_dups, save_dir_name, shuffle=False, split=False)

def find_and_save_junk_from_dataset_of_articles(article_dataset: Dataset, delimeter: str, junk_max_size: int):
    """Count junk snippets in dataset of articles, and add the counts to a dict, then save the dict as json file.

    Args:
        article_dataset (Dataset): The Dataset to look for junk in.
        delimeter (str): Character to split sequence on.
        junk_max_size (int): max length of the key saved in dict.
    """
    junk_counts: dict = find_junk_in_rows(article_dataset, columns=["title", "description"], delimeter=delimeter, junk_max_length=junk_max_size)

    # save dict
    delimeter_name = "custom"
    if delimeter == " - ": delimeter_name = "horizontal"
    if delimeter == "|":   delimeter_name = "vertical"
    save_file_name = "junk_found_%s_delimeter" % delimeter_name
    with open("./blobs/%s.json" % save_file_name, "w", encoding='utf-8') as f:
        json_record = json.dumps(junk_counts)
        f.write(json_record)

def remove_found_junk_from_articles_and_save(article_dataset):
    """Calls remove_junk_from_rows() with the "only_snippets" parameter set to True to avoid removing dates and html again.
    The "junk_snippets" parameter is extracted from the "junk_found_" json files created using find_junk_in_dataset_of_articles().
    Saves the cleaned dataset as HuggingFace dataset.
    """
    
    with open("./blobs/junk_found_horizontal_delimeter.json") as json_file:
        junk_snippets_hori = json.load(json_file)
        junk_snippets_hori_filtered = [snippet for snippet, count in junk_snippets_hori.items() if count > 32 and snippet not in [" - ", " -", "- ", "-"]]
    with open("./blobs/junk_found_vertical_delimeter.json") as json_file:
        junk_snippets_vert = json.load(json_file)
        junk_snippets_vert_filtered = [snippet for snippet, count in junk_snippets_vert.items() if count > 8 and snippet not in ["|", " |", "| ", " | "]]
    # since it is certain that the two dicts have no overlap (all keys are pre- or postfixed with a different symbol in each list), 
    # we can merge the list like this without worrying about duplicates
    junk_snippets = junk_snippets_hori_filtered + junk_snippets_vert_filtered

    # sorting the snippets, so long snippets are removed first, thereby avoiding a substrnig of one snippet being removed before the whole snippet
    # Example: "\||\| ABC" removed from "Hey | ABC" would become "Hey ABC", thereby leaving ABC, which is junk
    junk_snippets.sort(key=len, reverse=True)

    cleaned_dataset = remove_junk_from_rows(article_dataset, columns=["title", "description"], junk_snippets=junk_snippets, only_snippets=True)

    save_dataset_as_arrow(cleaned_dataset, DATASET_NODATES_NODUPS_CLEANED_DIR_NAME, shuffle=False, split=False)

def filter_dataset_of_articles_and_save(article_dataset):
    """Call remove_whole_junk_rows() on the cleaned dataset and save the new "filtered" dataset.
    """
    filtered_dataset: Dataset = remove_whole_junk_rows(dataset=article_dataset, columns=["title", "description"])
    save_dataset_as_arrow(filtered_dataset, FILTERED_DATASET_DIR_NAME, shuffle=False, split=False)

def create_labeled_dataset_multi(article_dataset):
    """Load the keywords for multi-label labeling and pass it to create_seq_class_dataset_mc_ml(), saving the resulting dataset.
    The extraction is performed on the cleaned and filtered dataset.
    """

    with open("./blobs/multi_class_keywords.json") as json_file:
       multi_class_keywords = json.load(json_file)
    multi_label_dataset = create_seq_class_dataset_mc_ml(multi_class_keywords, article_dataset, labeled_dataset_size=0.005)

    # assuming an even distribution of labels because of shuffling
    split_dataset = create_split_from_scratch(multi_label_dataset, shuffle=True, train_size=0.9)

    save_dataset_as_arrow(split_dataset, LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME, shuffle=False, split=False)

def create_labeled_dataset_multi_reselect_all(labeled_article_dataset):
    """This function is just a relabeling of the already labeled multi-class dataset, meant to adjust the distrubution of classes, because it went wrong the first time.

    Args:
        labeled_article_dataset (Dataset): A multi-labeled dataset with an uneven distribution
    """

    with open("./blobs/multi_class_keywords.json") as json_file:
       multi_class_keywords = json.load(json_file)

    # run the labeling three times, once for each split, making sure to pass the article_ids that are "off limits", so there is no overlap between splits.
    split_dataset: DatasetDict = DatasetDict()
    # using 80 percent of the already labeled data as base, because all classes can't reach the target size, if all data is used, due to the distribution being uneven
    split_dataset["train"] = create_seq_class_dataset_mc_ml_reselect(multi_class_keywords, labeled_article_dataset,      labeled_dataset_size=0.90*0.80)
    added_article_ids: dict = {k: None for k in split_dataset["train"][:]["new_article_id"]}
    split_dataset["test"] = create_seq_class_dataset_mc_ml_reselect(multi_class_keywords, labeled_article_dataset,       labeled_dataset_size=0.09*0.80, exclude_article_ids=added_article_ids)
    added_article_ids.update({k: None for k in split_dataset["test"][:]["new_article_id"]})
    split_dataset["validation"] = create_seq_class_dataset_mc_ml_reselect(multi_class_keywords, labeled_article_dataset, labeled_dataset_size=0.01*0.80, exclude_article_ids=added_article_ids)

    save_dataset_as_arrow(split_dataset, LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_DIR_NAME+"_test", shuffle=False, split=False)

def create_labeled_dataset_multi_reselect_2000(labeled_article_dataset: Dataset):
    """The same as create_labeled_dataset_multi_reselect_all() but only aiming for 2000 trinaing samples.

    Args:
        labeled_article_dataset (Dataset): A multi-labeled dataset with an uneven distribution
    """

    with open("./blobs/multi_class_keywords.json") as json_file:
       multi_class_keywords = json.load(json_file)

    total_size = labeled_article_dataset.num_rows

    # run the labeling three times, once for each split, making sure to pass the article_ids that are "off limits", so there is no overlap between splits.
    split_dataset: DatasetDict = DatasetDict()
    # aiming for 2000 total rows. labeled_dataset_size is the sum of labels, but one row can have multile labels, so it is set higher than 2000 to try to hit 2000 rows.
    split_dataset["train"] = create_seq_class_dataset_mc_ml_reselect(multi_class_keywords, labeled_article_dataset,      labeled_dataset_size=2400/total_size)
    added_article_ids: dict = {k: None for k in split_dataset["train"][:]["new_article_id"]}
    split_dataset["test"] = create_seq_class_dataset_mc_ml_reselect(multi_class_keywords, labeled_article_dataset,       labeled_dataset_size=240/total_size, exclude_article_ids=added_article_ids)
    added_article_ids.update({k: None for k in split_dataset["test"][:]["new_article_id"]})
    split_dataset["validation"] = create_seq_class_dataset_mc_ml_reselect(multi_class_keywords, labeled_article_dataset, labeled_dataset_size=24/total_size, exclude_article_ids=added_article_ids)
    added_article_ids.update({k: None for k in split_dataset["validation"][:]["new_article_id"]})

    save_dataset_as_arrow(split_dataset, LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME, shuffle=False, split=False)

def create_labeled_dataset_covid(article_dataset):
    """Load the keywords for covid-labeling and pass them to create_seq_class_dataset_covid(), saving the resulting dataset.
    The extraction is performed on the cleaned and filtered dataset.
    """
    with open("./blobs/covid_class_keywords.json") as json_file:
       covid_class_keywords = json.load(json_file)
    covid_label_dataset = create_seq_class_dataset_covid(covid_class_keywords, article_dataset, labeled_dataset_size=0.005)

    split_dataset = create_split_from_scratch(covid_label_dataset, shuffle=True, stratify_by_column="label", train_size=0.9)

    save_dataset_as_arrow(split_dataset, LABELED_COVID_CLASS_DATASET_DIR_NAME, shuffle=False, split=False)

def create_labeled_dataset_title_desc_match():
    """Loads one of the labeled sequence classification datasets, and creates mismatched datasets for each split. Removes unneeded columns. Saves the result.
    """
    # Any dataset containing the rows ["title", "description", "labels", "new_article_id", "meta"] and split into train-test-val
    labeled_multi_dataset = load_from_disk(LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME)

    # Removes some columns, since meta and ids no longer apply after shuffling, and labels because new labels are added.
    dataset = labeled_multi_dataset.remove_columns(["labels", "new_article_id", "meta"])

    # the disribution of true/false is automatically 50/50 in each split, because label_dataset_for_title_description_match makes it 50/50
    new_dataset = DatasetDict()
    new_dataset["train"] = label_dataset_for_title_description_match(dataset["train"])
    new_dataset["test"] = label_dataset_for_title_description_match(dataset["test"])
    new_dataset["validation"] = label_dataset_for_title_description_match(dataset["validation"])

    # don't split because it already is, but shuffle each split
    save_dataset_as_arrow(new_dataset, LABELED_TITLE_DESC_MATCH_DATASET_DIR_NAME, shuffle=True, split=False)

def main():
    """Runs all functions necessary to extract, clean label and repack the data to be ready for pre-training and fine-tuning
    """

    ### Initial data extraction and dataset creation
    extract_sciride_data_to_json()

    # Create dataset from jsonl files and remove initial duplicate articles
    create_and_save_dataset_of_unique_articles()

    ### Cleaning process
    # Remove dates (and some other general junk patterns)
    article_dataset: Dataset = load_from_disk(DATASET_DIR_NAME)
    remove_dates_and_save(article_dataset)

    # Remove duplicates from the dataset again, since new duplicates may have appeared after removing dates
    article_dataset: Dataset = load_from_disk(DATASET_NODATES_DIR_NAME)
    remove_duplicates_and_save(article_dataset, save_dir_name=DATASET_NODATES_NODUPS_DIR_NAME, id_column="new_article_id")

    # Look for junk associated with " - " and "|"
    article_dataset: Dataset = load_from_disk(DATASET_NODATES_NODUPS_DIR_NAME) 
    find_and_save_junk_from_dataset_of_articles(article_dataset, delimeter=" - ", junk_max_size=98)
    find_and_save_junk_from_dataset_of_articles(article_dataset, delimeter="|",   junk_max_size=76)
    
    # Remove all the found junk snippets that appeared {threshold} times in the data
    remove_found_junk_from_articles_and_save(article_dataset)

    # Remove whole articles classified as "junk"
    article_dataset: Dataset = load_from_disk(DATASET_NODATES_NODUPS_CLEANED_DIR_NAME) 
    filter_dataset_of_articles_and_save(article_dataset)

    # Remove duplicates again because the article contents was manipulated
    article_dataset: Dataset = load_from_disk(FILTERED_DATASET_DIR_NAME)
    remove_duplicates_and_save(article_dataset, save_dir_name=FILTERED_DATASET_NODUPS_DIR_NAME, id_column="new_article_id")
    
    ### Data extraction and labeling for fine tuning
    article_dataset: Dataset = load_from_disk(FILTERED_DATASET_NODUPS_DIR_NAME)

    # create_labeled_dataset_multi(article_dataset)
    labeled_multi: DatasetDict = merge_split_dataset(load_from_disk(LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_DIR_NAME))
    create_labeled_dataset_multi_reselect_all(labeled_multi)
    create_labeled_dataset_multi_reselect_2000(labeled_multi)
    
    create_labeled_dataset_covid(article_dataset)

    # Make sure there is no overlap between fine-tuning data and pre-training data
    remove_labeled_subset_from_full_data(article_dataset)

    create_labeled_dataset_title_desc_match()
    
    interactive_NER_labeler()
    save_NER_json_as_dataset()

    # create train-test split for NER based on names
    NER_data = load_from_disk(LABELED_NER_DATASET_DIR_NAME)
    create_and_save_split_for_NER(NER_data)
        

if __name__ == "__main__":
    main()