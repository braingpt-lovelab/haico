import re
import nltk
nltk.download('punkt')


def read_prompt_template(llm):
    prompt_template_fpath = f"prompt_templates/{llm.replace('/', '--')}.txt" # HF style
    print(f"Reading prompt template from {prompt_template_fpath}")
    with open(prompt_template_fpath, "r") as f:
        prompt_template = f.read()
    return prompt_template


def prepare_prompt_multiple_choice_harness(
        original_abstract, incorrect_abstract, prompt_template
    ):
    prompt_A = prompt_template.replace("<choice abstract>", original_abstract)
    prompt_B = prompt_template.replace("<choice abstract>", incorrect_abstract)
    return [prompt_A.strip(), prompt_B.strip()]


def extract_abstract_pair(abstract):
    # Find all occurrences of [[x,y]] in the line
    # e.g. ['rule,exemplar', 'exemplar,rule']
    matches = re.findall(r'\[\[(.*?)\]\]', abstract)
    # we iterate through all matches; 
    # and for each match, we replace
    # the match with the first item in the match.
    original_abstract = abstract
    incorrect_abstract = abstract
    for match in matches:
        # Split the match into two items
        items = match.split(',')
        # e.g. items = ['rule', 'exemplar']
        original_abstract = original_abstract.replace('[[' + match + ']]', items[0].strip())
        incorrect_abstract = incorrect_abstract.replace('[[' + match + ']]', items[1].strip())

    return original_abstract, incorrect_abstract


def extract_abstract_pair_isolated_sentences(abstract):
    """
    For the `eval_isolated_sentences` check, where 
    we produce both correct and incorrect collections of 
    sentences with choices (at least one) in them. In other words, 
    the abstracts are now sentences with choices in them.
    """
    # Split the text into sentences
    sentences = nltk.sent_tokenize(abstract)

    # Initialize lists to store parsed sentences
    original_collection = []
    incorrect_collection = []

    # Iterate through each sentence
    for sentence in sentences:
        # Check if the sentence contains [[ ]]
        if '[[' in sentence and ']]' in sentence:
            # Find all occurrences of [[x,y]] in the sentence
            matches = re.findall(r'\[\[(.*?)\]\]', sentence)

            # Iterate through all matches in the sentence
            correct_sentence = sentence
            incorrect_sentence = sentence
            for match in matches:
                # Split the match into two items
                items = match.split(',')
                
                # For each sentence, parse out the correct and incorrect versions
                correct_sentence = correct_sentence.replace('[[' + match + ']]', items[0].strip())
                incorrect_sentence = incorrect_sentence.replace('[[' + match + ']]', items[1].strip())

            # Add the parsed versions to the lists
            original_collection.append(correct_sentence)
            incorrect_collection.append(incorrect_sentence)

    return original_collection, incorrect_collection
