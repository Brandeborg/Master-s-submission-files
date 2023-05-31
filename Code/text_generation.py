import errno
from os import mkdir
from data_preparation import load_json, save_json
from transformers import pipeline

def create_generator(model_checkpoint, do_sample):
    """Given a model checkpoint use the HF pipeline to create a generator.

    Args:
        model_checkpoint (str): Checkpoint of the model passed to the pipeline function
        do_sample (bool): Whether to do sampling or not ("not" meaning be deterministic and greedy when auto-regressively choosing next word)

    Returns:
        Pipeline: A text-generation pipeline
    """
    return pipeline("text-generation", model=model_checkpoint, do_sample=do_sample, top_k = 5, top_p = 1.0, device=0)

def generate_text(prompt, generator, max_length, n_examples):
    """Given a prompt and a generator pipeline, generate the given (n_examples) amount of text seqeunces.

    Args:
        prompt (str): Prompt used to start text generation
        generator (Pipeline): Generator used to generate text
        n_examples (int): Number of generated examples per topic
        max_length (int): Max length of each generated seqeunce.

    Returns:
        (list): list of generated text strings
    """
    generated_text = generator(prompt, num_return_sequences=n_examples, max_length=max_length)
    generated_text_clean = [text["generated_text"] for text in generated_text]
    return generated_text_clean
    
def generate_text_from_topics(model_checkpoint, save_file, n_examples, max_length, path_prompt_topics):
    """Load prompt topics from JSON and use a generator, loaded from model checkpoint, to infer (generate) text.

    Args:
        model_checkpoint (str): Checkpoint of the model used for inference
        save_file (str): Name of the file to save the generated text to.
        n_examples (int): Number of generated examples per topic
        max_length (int): Max length of each generated seqeunce.
        path_prompt_topics (str): path to json file containing prompt topics.
    """
    generator_greedy = create_generator(model_checkpoint=model_checkpoint, do_sample=False)
    generator_sample = create_generator(model_checkpoint=model_checkpoint, do_sample=True)

    prompt_topics = load_json(path_prompt_topics)

    generated_text = {}

    for topic in prompt_topics:
        greedy = []
        sampling = []
        for prompt in prompt_topics[topic]["prompts"]:
            greedy.extend(generate_text(prompt, generator_greedy, n_examples=1, max_length=max_length))
            sampling.extend(generate_text(prompt, generator_sample, n_examples=n_examples, max_length=max_length))

        generated_text[topic] = {}
        generated_text[topic]["greedy"] = greedy
        generated_text[topic]["sampling"] = sampling

    save_json(generated_text, save_file)
    

def main():
    """Run text generation for different models
    """
    # create directory holding the results unless it exists
    try:
        mkdir("results")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    path_prompt_topics = "blobs/text_generation_prompt_topics.json"
    generate_text_from_topics("gpt2", save_file="results/generated_text_GenGPT.json", n_examples=100, max_length=50, path_prompt_topics=path_prompt_topics)
    generate_text_from_topics("models/gpt_sciride_news_rand_base/checkpoint-last", save_file="results/generated_text_NewsGPT.json", n_examples=100, max_length=50, path_prompt_topics=path_prompt_topics)
    generate_text_from_topics("models/gpt_sciride_news_gen_base/checkpoint-last", save_file="results/generated_text_GenNewsGPT.json", n_examples=100, max_length=50, path_prompt_topics=path_prompt_topics)


if __name__ == "__main__":
    main()