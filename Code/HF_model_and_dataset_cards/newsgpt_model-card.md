---
language: en

license: mit

widget:
- text: "COVID-19 is"
  example_title: "COVID"
- text: "The NBA will"
  example_title: "NBA"
- text: "Breaking news"
  example_title: "Breaking"
---

# NewsGPT

## Model Description
The model is similar to [gpt2](https://huggingface.co/gpt2) in that it shares its size, architecture, tokenizer algorithm and Causal Language Modeling objective. 
The model parameters of a [GPT2LMHeadModel](https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel) model were randomly initialized and pre-trained from scratch using a dataset consisting only of news.

## Training Data
The model's training data consists of ~13,000,000 English articles from ~90 outlets, which each consists of a headline (title) and a subheading (description). The articles were collected from the [Sciride News Mine](http://sciride.org/news.html), after which some additional cleaning was performed on the data, such as removing duplicate articles and removing repeated "outlet tags" appearing before or after headlines such as "| Daily Mail Online".

The cleaned dataset can be found on huggingface [here](https://huggingface.co/datasets/AndyReas/frontpage-news). 
The data was repacked before training, to avoid abrupt truncation, which altered the order of the data a bit but it is ultimately the same sentences.

## How to use
The model can be used with the HuggingFace pipeline like so:
```python
>>> from transformers import pipeline
>>> generator = pipeline('text-generation', model='andyreas/newsgpt')
>>> generator("COVID-19 is", max_length=50, num_return_sequences=2)

[{'generated_text': "COVID-19 is killing more people than the coronavirus. The number of people who have been infected has more than doubled in the past decade, according to a new analysis.The study of 2,000 people by the University of California.The study by"},
 {'generated_text': "COVID-19 is the worst thing to happen in Canada: A new study. A new study suggests that the COVID-19 pandemic has become the \"best thing to happen in Canada.\". But the pandemic has also been a long-term challenge for"}]
```
The model's config.json file includes default parameters for text-generation, which results in the same prompt producing different outputs. 
These can be overwritten to generate consistent outputs by setting "do_sample" = False, like so:

```python
>>> generator("COVID-19 is", do_sample=False)
```

or increase variance by increasing the amount of words considered during sampling, like so:

```python
>>> generator("COVID-19 is", do_sample=True, top_k=50)
```

## Training
Training ran for 1 epoch using a learning rate of 2e-5 and 50K warm-up steps out of ~800K total steps.

## Bias
Like any other model, NewsGPT is subject to bias according to the data it was trained on.