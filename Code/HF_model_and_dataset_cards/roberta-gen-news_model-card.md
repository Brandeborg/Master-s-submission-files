---
language: en

license: mit

widget:
- text: "Paris is the <mask> of France."
  example_title: "Paris is the <mask> of France."
- text: "The goal of life is <mask>."
  example_title: "The goal of life is <mask>."
---

# roberta-news

## Model Description
The model is [roberta-base](https://huggingface.co/roberta-base) fine-tuned to unmask news.

## Training Data
The model's training data consists of almost 13,000,000 English articles from ~90 outlets, which each consists of a headline (title) and a subheading (description). The articles were collected from the [Sciride News Mine](http://sciride.org/news.html), after which some additional cleaning was performed on the data, such as removing duplicate articles and removing repeated "outlet tags" appearing before or after headlines such as "| Daily Mail Online".

The cleaned dataset can be found on huggingface [here](https://huggingface.co/datasets/AndyReas/frontpage-news). roberta-gen-news was pre-trained on a large subset (12,928,029 / 13,118,041) of the linked dataset, after repacking the data a bit to avoid abrupt truncation.

## How to use
The model can be used with the HuggingFace pipeline like so:
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='andyreas/roberta-gen-news')
>>> print(unmasker("The weather forecast for <mask> is rain.", top_k=5))

[{'score': 0.06107175350189209, 
'token': 1083, 
'token_str': ' Friday', 
'sequence': 'The weather forecast for Friday is rain.'}, 
{'score': 0.04649643227458, 
'token': 1359, 
'token_str': ' Saturday', 
'sequence': 'The weather forecast for Saturday is rain.'
}, 
{'score': 0.04370906576514244, 
'token': 1772, 
'token_str': ' weekend', 
'sequence': 'The weather forecast for weekend is rain.'}, 
{'score': 0.04101456701755524, 
'token': 1133, 
'token_str': ' Wednesday', 
'sequence': 'The weather forecast for Wednesday is rain.'}, 
{'score': 0.03785591572523117, 
'token': 1234, 
'token_str': ' Sunday', 
'sequence': 'The weather forecast for Sunday is rain.'}]
```

## Training
Training ran for 1 epoch using a learning rate of 2e-6 and 50K warm-up steps out of ~800K total steps.

## Bias
Like any other model, roberta-gen-news is subject to bias according to the data it was trained on.