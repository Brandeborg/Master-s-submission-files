from typing import Callable
import errno
from os import mkdir
import os
import fine_tuning as ft, evaluation as ev, data_preparation as dp
from datasets import load_from_disk

# --- This file contains functions in charge of producing the fine-tuning evaluation results, meaning running fine-tuning and evaluation for different --- #
# --- tasks, multiple iterations, using different pre-trained models as the base and saving the results in JSON files                                  --- #

# --- LOAD ENV CONSTANTS FOR CONSISTENT FILE NAMES --- #
LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME = os.getenv("LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME")
LABELED_NER_DATASET_NAMESPLIT_DIR_NAME = os.getenv("LABELED_NER_DATASET_NAMESPLIT_DIR_NAME")

def worst_best_avg(metrics_list):
    """Find the best, worst and average metric in a list of metrics

    Args:
        metrics_list (dict): dict of lists of metrics (f1, accuracy, etc.)

    Returns:
        dict: dict of the averages
    """
    avg_metrics = {metric: {"worst": 1, "avg": 0, "best": 0} for metric in metrics_list[0]}
    
    metrics_list_len = len(metrics_list)

    for metrics in metrics_list:
        for metric in metrics:
            if metrics[metric] < avg_metrics[metric]["worst"]:
                avg_metrics[metric]["worst"] = metrics[metric]
            if metrics[metric] > avg_metrics[metric]["best"]:
                avg_metrics[metric]["best"] = metrics[metric]
            avg_metrics[metric]["avg"] += metrics[metric] / metrics_list_len

    return avg_metrics

### --- COVID --- ###
def FT_and_EV_covid(n_samples, pre_checkpoint, n_iter):
    """Run fine-tuning and evaluation using the labeled COVID-19 data for a number of iterations, saving the results at each iteration.

    Args:
        n_samples (int): Number of training samples
        pre_checkpoint (checkpoint): The pre-trained model used as base for the fine-tuning
        n_iter (int): Number of times to fine-tune and evaluate, using a different seed every time
    """
    # init dict which will be filled with evaluation results and eventually added to a JSON of other results
    ev_dict = {"n_iter": n_iter, "n_samples": n_samples, "base_model": pre_checkpoint, "model_raw": {"eval_raw": {}, "eval_sub": {}}, "model_sub": {"eval_raw": {}, "eval_sub": {}}}
    
    # fine tune without substituting keywords
    ft_raw_results = {"eval_raw": [], "eval_sub": []}
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 1/2")
        ft.load_data_and_fine_tune_covid(n_samples, pre_checkpoint, training_seed=i, sub=False)
        ft_raw_results["eval_raw"].append(ev.load_data_and_evaluate_covid(sub=False))
        ft_raw_results["eval_sub"].append(ev.load_data_and_evaluate_covid(sub=True))

    ev_dict["model_raw"]["eval_raw"]["individual"] = ft_raw_results["eval_raw"]
    ev_dict["model_raw"]["eval_raw"]["avg"] = worst_best_avg(ft_raw_results["eval_raw"])

    ev_dict["model_raw"]["eval_sub"]["individual"] = ft_raw_results["eval_sub"]
    ev_dict["model_raw"]["eval_sub"]["avg"] = worst_best_avg(ft_raw_results["eval_sub"])

    # fine tune after substituting keywords
    ft_sub_results = {"eval_raw": [], "eval_sub": []}
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 2/2")
        ft.load_data_and_fine_tune_covid(n_samples, pre_checkpoint, training_seed=i, sub=True)
        ft_sub_results["eval_raw"].append(ev.load_data_and_evaluate_covid(sub=False))
        ft_sub_results["eval_sub"].append(ev.load_data_and_evaluate_covid(sub=True))

    ev_dict["model_sub"]["eval_raw"]["individual"] = ft_sub_results["eval_raw"]
    ev_dict["model_sub"]["eval_raw"]["avg"] = worst_best_avg(ft_sub_results["eval_raw"])

    ev_dict["model_sub"]["eval_sub"]["individual"] = ft_sub_results["eval_sub"]
    ev_dict["model_sub"]["eval_sub"]["avg"] = worst_best_avg(ft_sub_results["eval_sub"])

    print(ft_raw_results)
    save_results(ev_dict, "single_label_classification", n_samples, pre_checkpoint)

### --- MULTI --- ###
def FT_and_EV_multi(n_samples, pre_checkpoint, n_iter):
    """Run fine-tuning and evaluation using the labeled politician data for a number of iterations, saving the results at each iteration.

    Args:
        n_samples (int): Number of training samples
        pre_checkpoint (checkpoint): The pre-trained model used as base for the fine-tuning
        n_iter (int): Number of times to fine-tune and evaluate, using a different seed every time
    """
    # Multi label is a special case because it was hard to hit 2000 exactly, so the dataset is loaded to check the actual amount of data
    actual_n_samples = load_from_disk(LABELED_MULTI_CLASS_MULTI_LABEL_DATASET_BETTER_2000_DIR_NAME)["train"].num_rows

    # init dict which will be filled with evaluation results and eventually added to a JSON of other results
    ev_dict = {"n_iter": n_iter, "n_samples": actual_n_samples, "base_model": pre_checkpoint, "model_raw": {"eval_raw": {}, "eval_sub": {}}, "model_sub": {"eval_raw": {}, "eval_sub": {}}}

    # fine tune without substituting keywords
    ft_raw_results = {"eval_raw": [], "eval_sub": []}
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 1/2")
        ft.load_data_and_fine_tune_multi(n_samples, pre_checkpoint, training_seed=i, sub=True)
        ft_raw_results["eval_raw"].append(ev.load_data_and_evaluate_multi(sub=False))
        ft_raw_results["eval_sub"].append(ev.load_data_and_evaluate_multi(sub=True))

    ev_dict["model_raw"]["eval_raw"]["individual"] = ft_raw_results["eval_raw"]
    ev_dict["model_raw"]["eval_raw"]["avg"] = worst_best_avg(ft_raw_results["eval_raw"])

    ev_dict["model_raw"]["eval_sub"]["individual"] = ft_raw_results["eval_sub"]
    ev_dict["model_raw"]["eval_sub"]["avg"] = worst_best_avg(ft_raw_results["eval_sub"])

    # fine tune after substituting keywords
    ft_sub_results = {"eval_raw": [], "eval_sub": []}
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 2/2")
        ft.load_data_and_fine_tune_multi(n_samples, pre_checkpoint, training_seed=i, sub=True)
        ft_sub_results["eval_raw"].append(ev.load_data_and_evaluate_multi(sub=False))
        ft_sub_results["eval_sub"].append(ev.load_data_and_evaluate_multi(sub=True))

    ev_dict["model_sub"]["eval_raw"]["individual"] = ft_sub_results["eval_raw"]
    ev_dict["model_sub"]["eval_raw"]["avg"] = worst_best_avg(ft_sub_results["eval_raw"])

    ev_dict["model_sub"]["eval_sub"]["individual"] = ft_sub_results["eval_sub"]
    ev_dict["model_sub"]["eval_sub"]["avg"] = worst_best_avg(ft_sub_results["eval_sub"])

    save_results(ev_dict, "multi_label_classification", n_samples, pre_checkpoint)

### --- NER --- ###
def FT_and_EV_NER(n_samples, pre_checkpoint, n_iter):
    """Run fine-tuning and evaluation using the labeled NER data for a number of iterations, saving the results at each iteration.

    Args:
        n_samples (int): Ignored because the NER dataset is small but needed for genericism. FT_and_EV_all_bases() takes a callback fucntion, it needs that parameter.
        pre_checkpoint (str): The checkpoint name of the pre-trained base model
        n_iter (int): How many times fine-tuning and evaluation in run with different initial parameters / weights
    """
    # NER is a special case because there is not enough data, so the dataset is loaded to check the actual amount of data
    actual_n_samples = load_from_disk(LABELED_NER_DATASET_NAMESPLIT_DIR_NAME)["train"].num_rows

    # init dict which will be filled with evaluation results and eventually added to a JSON of other results
    ev_dict = {"n_iter": n_iter, "n_samples": actual_n_samples, "n_epochs": 3, "base_model": pre_checkpoint, "model_gen": {}, "model_spec": {"PER": {}, "ORG": {}, "GPE": {}, "overall": {}}}

    # fine tune after generalizing labels
    ft_gen_results = []
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 1/2")
        ft.load_data_and_fine_tune_NER(pre_checkpoint, training_seed=i, gen=True)
        eval_results =  ev.load_data_and_evaluate_NER(gen=True)
        del eval_results["NE"]
        ft_gen_results.append(eval_results)

    ev_dict["model_gen"]["individual"] = ft_gen_results
    ev_dict["model_gen"]["avg"] = worst_best_avg(ft_gen_results)

    # fine tune without generalizing labels
    ft_spec_results = {"PER": [], "ORG": [], "GPE": [], "overall": []}
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 2/2")
        ft.load_data_and_fine_tune_NER(pre_checkpoint, training_seed=i, gen=False)
        eval_results = ev.load_data_and_evaluate_NER(gen=False)

        for NE_type in ["PER", "ORG", "GPE"]:
            del eval_results[NE_type]["number"]
            ft_spec_results[NE_type].append(eval_results[NE_type])
            del eval_results[NE_type]
        ft_spec_results["overall"].append(eval_results)

    for NE_type in ["PER", "ORG", "GPE", "overall"]:
        ev_dict["model_spec"][NE_type]["individual"] = ft_spec_results[NE_type]
        ev_dict["model_spec"][NE_type]["avg"] = worst_best_avg(ft_spec_results[NE_type])

    save_results(ev_dict, "token_classification", n_samples, pre_checkpoint)

### --- TITLE DESC MATCH --- ###
def FT_and_EV_title_desc_match(n_samples, pre_checkpoint, n_iter):
    """Run fine-tuning and evaluation using the labeled title-description match data for a number of iterations, saving the results at each iteration.

    Args:
        n_samples (int): Number of training samples
        pre_checkpoint (checkpoint): The pre-trained model used as base for the fine-tuning
        n_iter (int): Number of times to fine-tune and evaluate, using a different seed every time
    """
    ev_dict = {"n_iter": n_iter, "n_samples": n_samples, "base_model": pre_checkpoint, "eval": {}}

    ft_results = []
    for i in range(n_iter):
        print(i+1, "/", n_iter, " | 1/1")
        ft.load_data_and_fine_tune_title_desc_match(n_samples, pre_checkpoint, training_seed=i)
        ft_results.append(ev.load_data_and_evaluate_title_desc_match())

    ev_dict["eval"]["individual"]= ft_results
    ev_dict["eval"]["avg"] = worst_best_avg(ft_results)

    save_results(ev_dict, "title_desc_match_classification", n_samples, pre_checkpoint)

def save_results(result, task_name, n_samples, base_model):
    json_file_path = "results/ev_results_%s.json" % str(n_samples)

    # create directory holding the results unless it exists
    try:
        mkdir("results")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # load previously created results (other tasks)
    try:
        results = dp.load_json(json_file_path)
    except:
        results = {}

    # Add the new tasks to the existing
    try:
        results[task_name][base_model] = result
    except:
        results[task_name] = {}
        results[task_name][base_model] = result

    dp.save_json(dict=results, file_path=json_file_path)

def FT_and_EV_all_bases(FT_and_EV: Callable):
    """Run the passed ft-ev function using different bases and sample sizes

    Args:
        FT_and_EV (Callable): A function that fine-tunes a model, saves it, and then evaluates it using different data.
    """
    # number of samples and how many iterations
    sample_sizes = {(2000, 10)}#, (100, 10), (500, 10), (1000, 10), (5000, 5)}

    # main checkpoints
    bases = ["roberta-base",
            "models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2450000 (epoch 3)",
            "models/roberta_sciride_news_gen_base/checkpoint-last",
            None
             ]
    
    # other checkpoints
    bases_epochs = ["models/roberta_sciride_news_rand_base/checkpoints/checkpoint-1610000 (epoch 2)",
                    "models/roberta_sciride_news_rand_base/checkpoints/checkpoint-810000 (epoch 1)",
                    "models/roberta_sciride_news_rand_base/checkpoints/checkpoint-3240000 (epoch 4)"]
    
    bases_between = ["models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2850000",
                     "models/roberta_sciride_news_rand_base/checkpoints/checkpoint-2650000"]

    for n_samples, n_iter in sample_sizes:
        for base in bases:
            FT_and_EV(n_samples, base, n_iter)

def main():
    """Run FT_and_EV_all_bases for all tasks
    """
    import time 
    start = time.time()

    # Fine tuning and evaluation of Multi-Class, Multi-Label classification task (politicians)
    FT_and_EV_all_bases(FT_and_EV=FT_and_EV_multi)

    # Fine tuning and evaluation of Single-Label (binary) classification task (COVID-19)
    FT_and_EV_all_bases(FT_and_EV=FT_and_EV_covid)

    # Fine tuning and evaluation of double-sequence, Single-Label classification task (does title match description)
    FT_and_EV_all_bases(FT_and_EV=FT_and_EV_title_desc_match)

    # Fine tuning and evaluation of token classification task (NER)
    FT_and_EV_all_bases(FT_and_EV=FT_and_EV_NER)
    
    print("SEC: ", time.time() - start)

if __name__ == "__main__":
    main()