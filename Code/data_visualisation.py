import errno
import itertools
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import load_json

# --- This file runs all the functions which creates data visualisations for the report --- #

def create_plot(metrics: dict, columns: list, bar_colours: list, title: str, name: str,
                ylim=[0.0, 120], xlabel="Base model", ylabel="Scores",
                round_to_decimal=4, multiply_score=100, tick_rotation=0, 
                barlabel_rotation=0, barlabel_size=10, axtext=None):
    """Creates a bar chart from the given metrics and columns. Saves it.

    Args:
        metrics (dict): Values for the bars.
        columns (list): Column names
        bar_colours (list): ...
        title (str): Title of the final chart
        name (str): Name of the saved file
        ylim (list, optional): Span of y axis. Defaults to [0.0, 120].
        xlabel (str, optional): Label on the x asix. Defaults to "Base model".
        ylabel (str, optional): Label on the y asix. Defaults to "Scores".
        round_to_decimal (int, optional): How many decimals to round the metrics to. Defaults to 4.
        multiply_score (int, optional): How much to multiply the metrics with. Useful if original value is 0.0-1.0. Defaults to 100.
        tick_rotation (int, optional): Rotation of column names. Defaults to 0.
        barlabel_rotation (int, optional): Rotation of value displayed over bars. Defaults to 0.
        barlabel_size (int, optional): Font size of bar label. Defaults to 10.
        axtext (_type_, optional): Text going into optional textbox on the chart. Defaults to None.

    Raises:
        Exception: Length of metrics and bar colors need to be the same
    """
    
    if len(metrics) != len(bar_colours):
        raise Exception("There must be exacly as many colors as there are metrics")
    
    # create plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_ylim(ylim)
    index = np.arange(len(columns))
    opacity = 0.8

    bar_width = 1/(len(metrics)+1)
    multiplier = 0

    for metric, values in metrics.items():
        offset = bar_width * multiplier
        percentage_values = [round(value, round_to_decimal) * multiply_score for value in values]
        rects = plt.bar(index + offset, percentage_values, bar_width,
        alpha=opacity,
        color=bar_colours.pop(0),
        label=metric)
        ax.bar_label(rects, rotation=barlabel_rotation, padding=2, size=barlabel_size)

        if axtext != None:
            ax.text(0.795, 0.97, axtext, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, bbox = dict(boxstyle='round', facecolor='white', edgecolor="lightgray", alpha=0.5))

        multiplier += 1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width, columns, rotation=tick_rotation)
    plt.legend(loc="upper left", ncol = len(metrics))

    fig.tight_layout()

    plt.savefig("bar_charts/%s" % name)
    plt.close()

def create_plot_seq_class(task_results: dict, title: str, base_model_paths: dict, model="model_sub", eval="eval_sub", name="test_name"):
    """Given the task results for single-label classification (COVID-19) or multi-labell classification (politicians), plots the results
    in a bar chart for every base model.

    Args:
        task_results (dict): All the results for a given task, single_label_classification for instance, including the avg metrics.
        title (str): _description_
        base_model_paths (dict): _description_
        model (str, optional): Which data was used when fine-tuning the models, (keywords substituted or not). Either "model_sub" or "model_raw". Defaults to "model_sub".
        eval (str, optional):  Which data was used when evaluating the models, (keywords substituted or not). Either "eval_sub" or "eval_raw". Defaults to "eval_sub".
    """
    metrics = {"Accuracy (avg)":  [task_results[base_path][model][eval]["avg"]["accuracy"]["avg"]  for _, base_path in base_model_paths.items()],
               "Accuracy (best)": [task_results[base_path][model][eval]["avg"]["accuracy"]["best"] for _, base_path in base_model_paths.items()],
               "F1 (avg)":        [task_results[base_path][model][eval]["avg"]["f1"]["avg"]        for _, base_path in base_model_paths.items()],
               "F1 (best)":       [task_results[base_path][model][eval]["avg"]["f1"]["best"]       for _, base_path in base_model_paths.items()]}

    columns = base_model_paths.keys()

    bar_colours = ["forestgreen", "limegreen", "darkcyan", "cyan"]

    create_plot(columns=columns, 
                metrics=metrics, 
                bar_colours=bar_colours,
                tick_rotation=0, 
                title=title,
                name=name)

def create_plot_title_desc(task_results: dict, title: str, base_model_paths: dict, name="test_name"):
    """Very similar to create_plot_seq_class().
    """
    metrics = {"Accuracy (avg)":  [task_results[base_path]["eval"]["avg"]["accuracy"]["avg"]  for _, base_path in base_model_paths.items()],
               "Accuracy (best)": [task_results[base_path]["eval"]["avg"]["accuracy"]["best"] for _, base_path in base_model_paths.items()],
               "F1 (avg)":        [task_results[base_path]["eval"]["avg"]["f1"]["avg"]        for _, base_path in base_model_paths.items()],
               "F1 (best)":       [task_results[base_path]["eval"]["avg"]["f1"]["best"]       for _, base_path in base_model_paths.items()]}

    columns = base_model_paths.keys()

    bar_colours = ["forestgreen", "limegreen", "darkcyan", "cyan"]

    create_plot(columns=columns, 
                metrics=metrics, 
                bar_colours=bar_colours, 
                tick_rotation=0,
                title=title,
                name=name)

def create_plot_ner(task_results: dict, title: str, base_model_paths: dict, model="model_spec", NE_subtype="overall", name="test_name"):
    """Similar to create_plot_seq_class() but for NER task.

    Args:
        ...
        model (str, optional): Which data was used when fine-tuning and evaluating the model (with NE subtypes or not)
        NE_subtype (str, optional):  Which NE subtype to report. Defaults to "overall".
    """
    bar_colours = ["forestgreen", "limegreen", "darkcyan", "cyan"]

    if model == "model_gen":
        metrics = {
                "Accuracy (avg)":  [task_results[base_path][model]["avg"]["overall_accuracy"]["avg"]  for _, base_path in base_model_paths.items()],
                "Accuracy (best)": [task_results[base_path][model]["avg"]["overall_accuracy"]["best"] for _, base_path in base_model_paths.items()],
                "F1 (avg)":        [task_results[base_path][model]["avg"]["overall_f1"]["avg"]        for _, base_path in base_model_paths.items()],
                "F1 (best)":       [task_results[base_path][model]["avg"]["overall_f1"]["best"]       for _, base_path in base_model_paths.items()]}
        
    if model == "model_spec":
        if NE_subtype == "overall":
            metrics = {
                    "Accuracy (avg)":  [task_results[base_path][model][NE_subtype]["avg"]["overall_accuracy"]["avg"]  for _, base_path in base_model_paths.items()],
                    "Accuracy (best)": [task_results[base_path][model][NE_subtype]["avg"]["overall_accuracy"]["best"] for _, base_path in base_model_paths.items()],
                    "F1 (avg)":        [task_results[base_path][model][NE_subtype]["avg"]["overall_f1"]["avg"]        for _, base_path in base_model_paths.items()],
                    "F1 (best)":       [task_results[base_path][model][NE_subtype]["avg"]["overall_f1"]["best"]       for _, base_path in base_model_paths.items()]}
        else:
            metrics = {
                    "F1 (avg)":        [task_results[base_path][model][NE_subtype]["avg"]["f1"]["avg"]        for _, base_path in base_model_paths.items()],
                    "F1 (best)":       [task_results[base_path][model][NE_subtype]["avg"]["f1"]["best"]       for _, base_path in base_model_paths.items()]}
            bar_colours = ["darkcyan", "cyan"]

    columns = base_model_paths.keys()

    create_plot(columns=columns, 
                metrics=metrics, 
                bar_colours=bar_colours, 
                tick_rotation=0,
                title=title,
                name=name)

### text analysis

def create_plot_sentiment_per_topic(task_results: dict, topic: str, title: str, name="test_name", ylim=[-120, 120]):
    """Visualise the sentiment of a single topic, including avg pos, avg neg and avg overall. 

    Args:
        ...
        topic (str): Which topic (eg. Donald Trump) to show results for
        ...
    """
    bar_colours = ["limegreen", "silver", "crimson"]

    sentiment_scores = {"average positives": [task_results[base_result][topic]["avg_pos"] for base_result in task_results],
                        "average all":       [task_results[base_result][topic]["avg_all"] for base_result in task_results],
                        "average negatives": [task_results[base_result][topic]["avg_neg"] * -1 for base_result in task_results]}
    
    columns = task_results.keys()

    create_plot(columns=columns, 
                metrics=sentiment_scores, 
                bar_colours=bar_colours, 
                title=title,
                tick_rotation=0,
                ylim=ylim,
                ylabel="Average sentiment scores",
                name=name)
    

def create_plot_sentiment_all_topics(task_results: dict, title: str, name="test_name", ylim=[-120, 120]):
    """Visualize sentiment of all topics in task_results but only overall avg.
    """
    bar_colours = ["limegreen", "cyan", "yellow"]

    sentiment_scores = {}

    # restructure results so base models will be legends and avg sentiments for each topic will be bars
    for base_model in task_results:
        avgs = [v["avg_all"] for k, v in task_results[base_model].items()]
        sentiment_scores[base_model] = avgs
    
    # extract topics from one of the task results
    columns = list(task_results.values())[0].keys()

    create_plot(columns=columns, 
                metrics=sentiment_scores, 
                bar_colours=bar_colours, 
                title=title,
                ylim=ylim,
                xlabel="Topic",
                ylabel="Average sentiment score",
                tick_rotation=25,
                barlabel_rotation=90,
                name=name)

def create_plot_tf(task_results: dict, title: str, name="test_name", topic="All", ngram=1, ylim=[0,100], ylabel="Score"):
    """Visualize term frequencies as bar charts.

    Args:
        ...
        ngram (int, optional): Whether to use terms with 1, 2 og 3 "words". Defaults to 1.
        ...
    """
    bar_colours = ["limegreen", "cyan", "yellow"]

    tfidf_scores = {}

    # extract all feature words
    all_features = {}
    for base_model in task_results:
        for feature in task_results[base_model][topic][str(ngram)]:
            all_features[feature] = 0

    # restructure results so each base_model has list of scores per feature
    for base_model in task_results:
        all_features_copy = all_features.copy()
        # extract all scores from a specific topic from chosen ngram count
        base_tfidf_scores = task_results[base_model][topic][str(ngram)]

        # fill out the feature scores for each base model where scores are available, leaving the rest 0
        for feature in base_tfidf_scores:
            all_features_copy[feature] = base_tfidf_scores[feature]
        
        # extract only the scores, discarding feature keys
        tfidf_scores[base_model] = list(all_features_copy.values())

    # extract topics from one of the task results
    columns = list(all_features.keys())

    ## jaccard
    jaccard_indices = jaccard_index_all(task_results=task_results, topic=topic, ngram=ngram)
    axtext = "----------- Jaccard Index -----------\n\n"
    for key, val in jaccard_indices.items():
        axtext += "%s: %s\n" % (key, val)

    # remove last newline
    axtext = axtext[:-1]

    create_plot(columns=columns, 
                metrics=tfidf_scores, 
                bar_colours=bar_colours, 
                title=title,
                ylim=ylim,
                xlabel="Word",
                ylabel=ylabel,
                round_to_decimal=2,
                multiply_score=1,
                tick_rotation=90,
                barlabel_rotation=90,
                barlabel_size=6,
                axtext=axtext,
                name=name)

    ## table
    create_tf_table(task_results=task_results, topic=topic, ngram=ngram, name=name)

def jaccard_index(set1: set, set2: set):
    """Compute jaccard index between two sets.

    Returns:
        float: The jaccard index multiplied by 100
    """
    jaccard_index = len(set1.intersection(set2)) / len(set1.union(set2))
    return round(jaccard_index * 100, 2) 

def jaccard_index_all(task_results: dict, topic="All", ngram=2):
    """Create jaccard indices for all pairs of model results.

    Returns:
        dict: {model1_model2: [jaccard index]}
    """
    ngram = str(ngram)
    keys = list(task_results.keys())
    
    pairs = list(itertools.combinations(keys, 2))

    jaccard_indices = {
        "%s_%s" % (e1, e2): jaccard_index(
        set(task_results[e1][topic][ngram].keys()), 
        set(task_results[e2][topic][ngram].keys())) 
        for e1, e2 in pairs
        }
    
    return jaccard_indices

def create_tf_table_dict(task_results, topic="All", ngram=2):
    """create dictionary to serve as the basis for a table that should look like this:

    term / model       GenGPT        NewsGPT        GenNewsGPT
       term1        score (rank)   score (rank)    score (rank)
       term2           2.1 (1)       2.4 (0)          1.8 (2)

    The dictionary will look like this:
    {
        "Term":       ["Term 1",             "Term 2"], 
        "GenGPT":     ["Score (rank)", "Score (rank)"], 
        "NewsGPT":    ["Score (rank)", "Score (rank)"], 
        "GenNewsGPT": ["Score (rank)", "Score (rank)"]
    }

    Args:
        task_results (_type_): _description_
        topic (str, optional): _description_. Defaults to "All".
        ngram (int, optional): _description_. Defaults to 2.

    Returns:
        dict:
    """
    ngram = str(ngram)

    term_ranks = {}
    for base_model in task_results:
        model_term_scores = task_results[base_model][topic][ngram]
        for i, term in enumerate(model_term_scores):
            if term not in term_ranks:
                term_ranks[term] = {}
            
            term_ranks[term][base_model] = "%s (%s)" % (round(model_term_scores[term], 2), i+1)

    table_dict = {"Term": (term_ranks.keys())}
    for model in task_results:
        table_dict[model] = []

    for model in task_results:
        for term in list(term_ranks.values()):
            try: 
                table_value = term[model]
            except:
                table_value = ""
            table_dict[model].append(table_value)


    return table_dict

def create_tf_table(task_results, topic="All", ngram=2, name=""):
    """Create table version of the top-20 lists.
    """
    from pandas import DataFrame

    #define figure and axes
    fig, ax = plt.subplots(figsize=(8,12), frameon=False)

    #hide the axes
    fig.patch.set_visible(False)
    ax.axis('tight')
    ax.axis('off')
    ax.margins(x=0)

    table_dict = create_tf_table_dict(task_results, topic=topic, ngram=ngram)
    df = DataFrame.from_dict(table_dict)

    #create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')

    #display table
    fig.tight_layout()
    plt.autoscale(tight=True)
    plt.savefig(("bar_charts/%s_table" % name), bbox_inches='tight', pad_inches=0.0)
    plt.close('all')


def main():
    """Pass several combinations of the data to the plotting functions.
    """
    # create directory holding the results unless it exists
    try:
        mkdir("bar_charts")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ### FINE-TUNING ###

    all_ft_ev_results = load_json("results/ev_results_%s.json" % 2000)
    n_samples_covid = all_ft_ev_results["single_label_classification"]["roberta-base"]["n_samples"]

    base_model_paths = {"roberta-gen": "roberta-base", 
                        "roberta-news (e3)": "models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2450000 (epoch 3)", 
                        "roberta-gen-news (e1)": "models/roberta_sciride_news_gen_base/checkpoint-last", 
                        "Not pre-trained": "null"}

    ## COVID CLASS
    create_plot_seq_class(all_ft_ev_results["single_label_classification"], 
                        title="Metrics for single-label sequence classification (COVID-19) (no keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_covid, 
                        base_model_paths=base_model_paths,
                        model="model_raw",
                        eval="eval_raw",
                        name="single_label_modelraw_evalraw_%s" % n_samples_covid)
    
    create_plot_seq_class(all_ft_ev_results["single_label_classification"], 
                        "Metrics for single-label sequence classification (COVID-19) (test keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_covid, 
                        base_model_paths=base_model_paths,
                        model="model_raw",
                        eval="eval_sub",
                        name="single_label_modelraw_evalsub_%s" % n_samples_covid)
    
    create_plot_seq_class(all_ft_ev_results["single_label_classification"], 
                        "Metrics for single-label sequence classification (COVID-19) (train keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_covid, 
                        base_model_paths=base_model_paths,
                        model="model_sub",
                        eval="eval_raw",
                        name="single_label_modelsub_evalraw_%s" % n_samples_covid)
    
    create_plot_seq_class(all_ft_ev_results["single_label_classification"], 
                        "Metrics for single-label sequence classification (COVID-19) (train and test keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_covid, 
                        base_model_paths=base_model_paths,
                        model="model_sub",
                        eval="eval_sub",
                        name="single_label_modelsub_evalsub_%s" % n_samples_covid)
    
    # MULTI CLASS
    n_samples_multi = all_ft_ev_results["multi_label_classification"]["roberta-base"]["n_samples"]
    create_plot_seq_class(all_ft_ev_results["multi_label_classification"], 
                        "Metrics for multi-label sequence classification (politicians) (no keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_multi, 
                        base_model_paths=base_model_paths,
                        model="model_raw",
                        eval="eval_raw",
                        name="multi_label_modelraw_evalraw_%s" % n_samples_multi)
    
    create_plot_seq_class(all_ft_ev_results["multi_label_classification"], 
                        "Metrics for multi-label sequence classification (politicians) (test keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_multi,
                        base_model_paths=base_model_paths,
                        model="model_raw",
                        eval="eval_sub",
                        name="multi_label_modelraw_evalsub_%s" % n_samples_multi)
    
    create_plot_seq_class(all_ft_ev_results["multi_label_classification"], 
                        "Metrics for multi-label sequence classification (politicians) (train keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_multi, 
                        base_model_paths=base_model_paths,
                        model="model_sub",
                        eval="eval_raw",
                        name="multi_label_modelsub_evalraw_%s" % n_samples_multi)
    
    create_plot_seq_class(all_ft_ev_results["multi_label_classification"], 
                        "Metrics for multi-label sequence classification (politicians) (train and test keywords masked) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_multi, 
                        base_model_paths=base_model_paths,
                        model="model_sub",
                        eval="eval_sub",
                        name="multi_label_modelsub_evalsub_%s" % n_samples_multi)
    

    # TITLE DESC CLASS
    n_samples_title_desc = all_ft_ev_results["title_desc_match_classification"]["roberta-base"]["n_samples"]
    create_plot_title_desc(all_ft_ev_results["title_desc_match_classification"],
                        "Metrics for single-label sequence classification (title-description match) per base model.\nn_train_samples = %s\nn_test_samples = 200" % n_samples_multi,
                        base_model_paths=base_model_paths,
                        name="title_desc_match_%s" % n_samples_title_desc)
    
    # TOKEN CLASS
    n_samples_ner = all_ft_ev_results["token_classification"]["roberta-base"]["n_samples"]
    create_plot_ner(all_ft_ev_results["token_classification"],
                    title="Metrics for token classification (NER) without NE subtypes per base model. \nn_train_samples=%s \nn_test_samples = 34" % n_samples_ner,
                    base_model_paths=base_model_paths,
                    model="model_gen",
                    name="token_classification_gen_%s" % n_samples_ner)
    
    create_plot_ner(all_ft_ev_results["token_classification"],
                    title="Metrics for token classification (NER) with NE subtypes per base model. \nn_train_samples=%s \nn_test_samples = 34" % n_samples_ner,
                    base_model_paths=base_model_paths,
                    model="model_spec",
                    NE_subtype="overall",
                    name="token_classification_spec_overall_%s" % n_samples_ner)
    
    create_plot_ner(all_ft_ev_results["token_classification"],
                    title="Metrics for token classification (NER) with NE subtypes (only PER) per base model. \nn_train_samples<%s \nn_test_samples < 34" % n_samples_ner,
                    base_model_paths=base_model_paths,
                    model="model_spec",
                    NE_subtype="PER",
                    name="token_classification_spec_PER_%s" % n_samples_ner)
    
    
    ### TEXT-ANALYSIS
    ## Sentiment
    file_names = {"GenGPT": "results/generated_text_GenGPT_sentiment.json",
                  "NewsGPT": "results/generated_text_NewsGPT_sentiment.json",
                  "GenNewsGPT": "results/generated_text_GenNewsGPT_sentiment.json"}
    
    sentiment_results = {k: load_json(v) for k, v in file_names.items()}

    GPEs = ["The US", "Russia", "China", "North Korea", "South Korea", "The EU", "The UK", "Denmark"]
    sentiment_results_GPEs = {k: {inner_k: inner_v for inner_k, inner_v in v.items() if inner_k in GPEs} for k, v in sentiment_results.items()}

    create_plot_sentiment_per_topic(sentiment_results, topic="All", 
                                    title="Average sentiment scores of generated text for all topics per base model.\nn_seq = %s" % (4*100*len(sentiment_results["GenGPT"])), 
                                    name="avg_sentiments_all",
                                    ylim=[-150,150])
    create_plot_sentiment_all_topics(sentiment_results_GPEs, 
                                     title="Average sentiment score of generated text for each topic (within GPEs) per base model.\nn_seq_per_topic = %s" % (4*100), 
                                     name="avg_sentiment_per_topic_GPE",
                                     ylim=[-130,130])

    world_leaders = ["Donald Trump", "Joe Biden", "Angela Merkel", "Vladimir Putin", "Kim Jong-Un"]
    sentiment_results_world_leaders = {k: {inner_k: inner_v for inner_k, inner_v in v.items() if inner_k in world_leaders} for k, v in sentiment_results.items()}
    create_plot_sentiment_all_topics(sentiment_results_world_leaders, 
                                     title="Average sentiment score of generated text for each topic (within world leaders) per base model.\nn_seq_per_topic = %s" % (4*100),
                                     name="avg_sentiment_per_topic_world_leaders",
                                     ylim=[-130,130])

    sentiment_results_remaining_topics = {k: {inner_k: inner_v for inner_k, inner_v in v.items() if inner_k not in GPEs+world_leaders} for k, v in sentiment_results.items()}
    create_plot_sentiment_all_topics(sentiment_results_remaining_topics, 
                                     title="Average sentiment score of generated text for each topic (within the remaining) topics per base model.\nn_seq_per_topic = %s" % (4*100),
                                     name="avg_sentiment_per_topic_remaining",
                                     ylim=[-130,130])

    ## TF-IDF
    file_names = {"GenGPT": "results/generated_text_GenGPT_tfidf.json",
                  "NewsGPT": "results/generated_text_NewsGPT_tfidf.json",
                  "GenNewsGPT": "results/generated_text_GenNewsGPT_tfidf.json"}
    
    tfidf_results = {k: load_json(v) for k, v in file_names.items()}

    # unigams
    create_plot_tf(tfidf_results, topic="All", 
                      title="Top 20 (TF-IDF) unigrams used in generated text for all topics per base model.\nn_seq = %s" % (4*100*len(sentiment_results["GenGPT"])), 
                      name="tfidf_all_uni",
                      ngram=1,
                      ylim=[0,320],
                      ylabel="Sum of TF-IDF scores")
    create_plot_tf(tfidf_results, topic="Climate change", 
                      title="Top 20 (TF-IDF) unigrams used in generated text for one topic (Climate change) per base model.\nn_seq = %s" % (4*100), 
                      name="tfidf_cliamte_change_uni",
                      ngram=1,
                      ylim=[0,65],
                      ylabel="Sum of TF-IDF scores")
    create_plot_tf(tfidf_results, topic="The EU", 
                      title="Top 20 (TF-IDF) unigrams used in generated text for one topic (The EU) per base model.\nn_seq = %s" % (4*100), 
                      name="tfidf_the_EU_uni",
                      ngram=1,
                      ylim=[0,65],
                      ylabel="Sum of TF-IDF scores")
    create_plot_tf(tfidf_results, topic="Kim Jong-Un", 
                      title="Top 20 (TF-IDF) unigrams used in generated text for one topic (Kim Jong-Un) per base model.\nn_seq = %s" % (4*100), 
                      name="tfidf_kim_jong-un_uni",
                      ngram=1,
                      ylim=[0,65],
                      ylabel="Sum of TF-IDF scores")
    
    # bigrams
    create_plot_tf(tfidf_results, topic="All", 
                      title="Top 20 (TF-IDF) bigrams used in generated text for all topics per base model.\nn_seq = %s" % (4*100*len(sentiment_results["GenGPT"])), 
                      name="tfidf_all_bi",
                      ngram=2,
                      ylim=[0,160],
                      ylabel="Sum of TF-IDF scores")
    create_plot_tf(tfidf_results, topic="Climate change", 
                      title="Top 20 (TF-IDF) bigrams used in generated text for one topic (Climate change) per base model.\nn_seq = %s" % (4*100), 
                      name="tfidf_cliamte_change_bi",
                      ngram=2,
                      ylim=[0,45],
                      ylabel="Sum of TF-IDF scores")
    create_plot_tf(tfidf_results, topic="The EU", 
                      title="Top 20 (TF-IDF) bigrams used in generated text for one topic (The EU) per base model.\nn_seq = %s" % (4*100), 
                      name="tfidf_the_EU_bi",
                      ngram=2,
                      ylim=[0,45],
                      ylabel="Sum of TF-IDF scores")
    create_plot_tf(tfidf_results, topic="Kim Jong-Un", 
                      title="Top 20 (TF-IDF) bigrams used in generated text for one topic (Kim Jong-Un) per base model.\nn_seq = %s" % (4*100), 
                      name="tfidf_kim_jong-un_bi",
                      ngram=2,
                      ylim=[0,45],
                      ylabel="Sum of TF-IDF scores")

    ## TF
    file_names = {"GenGPT": "results/generated_text_GenGPT_term_freq.json",
                  "NewsGPT": "results/generated_text_NewsGPT_term_freq.json",
                  "GenNewsGPT": "results/generated_text_GenNewsGPT_term_freq.json"}
    
    tf_results = {k: load_json(v) for k, v in file_names.items()}

    # unigrams
    create_plot_tf(tf_results, topic="All", 
                      title="Top 20 (TF) unigrams used in generated text for all topics per base model.\nn_seq = %s" % (4*100*len(sentiment_results["GenGPT"])), 
                      name="tf_all_uni",
                      ylim=[0,4300],
                      ngram=1,
                      ylabel="Sum of term frequencies")
    create_plot_tf(tf_results, topic="Climate change", 
                      title="Top 20 (TF) unigrams used in generated text for one topic (Climate change) per base model.\nn_seq = %s" % (4*100), 
                      name="tf_climate_change_uni",
                      ylim=[0,700],
                      ngram=1,
                      ylabel="Sum of term frequencies")
    create_plot_tf(tf_results, topic="The EU", 
                      title="Top 20 (TF) unigrams used in generated text for one topic (The EU) per base model.\nn_seq = %s" % (4*100), 
                      name="tf_the_EU_uni",
                      ylim=[0,700],
                      ngram=1,
                      ylabel="Sum of term frequencies")
    create_plot_tf(tf_results, topic="Kim Jong-Un", 
                      title="Top 20 (TF) unigrams used in generated text for one topic (Kim Jong-Un) per base model.\nn_seq = %s" % (4*100), 
                      name="tf_kim_jong-un_uni",
                      ylim=[0,700],
                      ngram=1,
                      ylabel="Sum of term frequencies")
    
    # bigrams
    create_plot_tf(tf_results, topic="All", 
                      title="Top 20 (TF) bigrams used in generated text for all topics per base model.\nn_seq = %s" % (4*100*len(sentiment_results["GenGPT"])), 
                      name="tf_all_bi",
                      ylim=[0,1200],
                      ngram=2,
                      ylabel="Sum of term frequencies")
    create_plot_tf(tf_results, topic="Climate change", 
                      title="Top 20 (TF) bigrams used in generated text for one topic (Climate change) per base model.\nn_seq = %s" % (4*100), 
                      name="tf_climate_change_bi",
                      ylim=[0,350],
                      ngram=2,
                      ylabel="Sum of term frequencies")
    create_plot_tf(tf_results, topic="The EU", 
                      title="Top 20 (TF) bigrams used in generated text for one topic (The EU) per base model.\nn_seq = %s" % (4*100), 
                      name="tf_the_EU_bi",
                      ylim=[0,350],
                      ngram=2,
                      ylabel="Sum of term frequencies")
    create_plot_tf(tf_results, topic="Kim Jong-Un", 
                      title="Top 20 (TF) bigrams used in generated text for one topic (Kim Jong-Un) per base model.\nn_seq = %s" % (4*100), 
                      name="tf_kim_jong-un_bi",
                      ylim=[0,350],
                      ngram=2,
                      ylabel="Sum of term frequencies")


if __name__ == "__main__":
    main()