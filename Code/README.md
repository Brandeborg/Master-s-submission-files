# Prepare Python virtual environment (windows)
## install
```cmd
pip3 install virtualenv
```

## create
```cmd
python3 -m venv .venv
```

## Activate 
```cmd
.\.venv\Scripts\activate
```

## Deactivate
```cmd
.\.venv\Scripts\deactivate
```

## install python modules
```cmd
pip3 install -r requirements.txt
```
note that requirements.txt was manually created, adding libraries whenever they were installed, and does not include version numbers, so the above command may install newer versions of modules than what was used in the project.
requirements_from_venv.txt was extracted from the venv, so it includes version numbers.

## install pytorch modules with cuda available
```cmd
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
Might want a different version:
https://pytorch.org/get-started/locally/


# Reproduce the results
## Get the raw news data
1. Go to http://sciride.org/news.html and download the "Processed Pages" or follow this download url https://news-mine.s3.eu-west-2.amazonaws.com/processed.tar.gz
2. Extract the .tar.gz file and place the resulting "release" directory in the "root" directory of this project

Alternatively, navigate to the "root" directory of this project and run (in wsl)
```bash
# wget -c https://news-mine.s3.eu-west-2.amazonaws.com/processed.tar.gz -O - | tar -xz
```

## Prepare the data for training
1. Start the virtual environment using
```cmd
.\.venv\Scripts\activate
```
2. Run all functions in the main() of data_preparation.py. Maybe do it a few functions at a time if memory is an issue.

## Run pre-training
1. Run the functions in the main() of language_modeling.py. pretrain_gpt2_news() and pretrain_roberta_news() was trained with learning_rate = 2e-5 while fine_tune_gpt2_on_news() and fine_tune_roberta_on_news() was trained with learning_rate = 2e-6. These can be adjusted in TrainingArguments in train_language_model() in language_modeling.py. The numbers were chosen after some experimentation, there may be better values.
2. Note that the above step takes approximately one full month using a GTX 1070 GPU. It's probably not a good idea to actually run it again. The models are available at https://huggingface.co/AndyReas. Instead of running pre-training again, an option is to download those and place them in an identical folder/name structure to where the trained models would end up. See .env and train_language_model() to find out what models are called.

## Run fine-tuning
1. Run the functions in the main() of model_comparison.py to run fine-tuning and evaluation.

## Run text generation
1. Run the functions in the main() of text_generation.py to generate a corpus of generated text. Note that a degree of randomness is used in the decoding strategy of the text generation, so the final results may be a bit different.

## Run text analysis
1. Run the functions in the main() of text_analysis.py to run term frequency and sentiment analysis.

## Visualise the results
1. Run the functions in the main() of data_visualisation.py to produce the bar charts visualising the results.


Note:
/blobs/AvailableOutlets.txt was provided by Konrad Krawczyk, taken from the sciride project.
AvailableOutlets.json and english_outlets.json was derived from that file.