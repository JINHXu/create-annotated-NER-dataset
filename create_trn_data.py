'''
Author: Jinghua Xu
Hornor Code: I pledge that the code repreesent my own work.
Description: Create training data for NER from reddit posts/comments using patterns and SpaCy. Save training data to a file.
'''

import spacy
from spacy.matcher import Matcher
# from spacy.pipeline import EntityRuler
import json
import sys
import random

# Token-based Matching https://spacy.io/usage/rule-based-matching

if len(sys.argv) < 2:
    sys.exit('Too few arguments, please speciify the input file')

filename = sys.argv[1]
# load the reddit data
with open(filename, 'r', encoding="utf-8") as f:
    data = [json.loads(line) for line in f.readlines()]
    redditJson = json.loads(json.dumps(data))


# extract the texts from json
# key could be 'selftext', 'body'
texts = list()
for entry in redditJson:
    if 'selftext' in entry:
        texts.append(entry['selftext'])
    if 'body' in entry:
        texts.append(entry['body'])


#################################################
# Select n random elements from texts
# Use a seed so that the same samples are selected on subsequent runs of the program
n = 300
seed = 27
random.seed(seed)
randomTexts = random.sample(texts, n)
#################################################


# load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# patterns

# link to commonly used drugs: https://www.drugabuse.gov/sites/default/files/Commonly-Used-Drugs-Charts_final_June_2020_optimized.pdf
# list of common drug names all in lower case
common_drug_names = ['ritalin', 'adderall', 'amphetamine', 'alcohol', 'ayahuasca', 'cocaine', 'dimethyltriptamine', 'dmt', 'gamma-hydroxybutyrate', 'ghb', 'hallucinogens', 'heroin', 'ketamine', 'khat', 'kratom', 'lsd', 'marijuana', 'ecstasy', 'mescaline', 'anabolic',
                     'mdma', 'peyote', 'methamphetamine', 'dextromethorphan', 'dxm', 'loperamide', 'pcp', 'opioids', 'psilocybin', 'flunitrazepam', 'rohypnol', 'salvia', 'anabolic', 'steroids', 'tabacco', 'nicotine']

# a pattern to match common drug names
# upper and lower case are always not diffeerentiated in reddit posts/comments, for example, 'LSD' often show as 'lsd'
# the pattern will match all drug names whose lower case form is in the list common_drug_names
pattern0 = [
    {
        'LOWER': {'IN': common_drug_names},
        'POS': 'NOUN'
    }
]


# unlike ingredients, drug entities rarely show in NOUN-NOUN or ADJ-NOUN pairs, most drug names are single-token, i.e. rarely any durg name is a phrase, for example there is no "heroin cocaine"
# though some street names of drugs may appear as phrase such as "Vitamine K"
# according to experiments, adding street names generally improves the model performance

# vatamine k, lady k, special k
pattern1 = [{"LOWER": {'IN': ["vitamine", "lady", "special"]},
             'POS': 'NOUN'}, {"LOWER": "k"}]
# laughing gas
pattern2 = [{"LOWER": "laughing"}, {"LOWER": "gas"}]
# Cat Valium
pattern3 = [{"LOWER": "cat"}, {"LOWER": "valium"}]
# Date Rape Drug
pattern4 = [{"LOWER": "date"}, {"LOWER": "rape"}, {"LOWER": "drug"}]
# purple passion
pattern5 = [{"LOWER": "purple"}, {"LOWER": "passion"}]
# Forget-Me Pill
pattern6 = [{"LOWER": "forget-me"}, {"LOWER": "pill"}]
# # Sewage Fruit
pattern7 = [{"LOWER": "sewage"}, {"LOWER": "fruit"}]
# Little Smoke
pattern8 = [{"LOWER": "little"}, {"LOWER": "smoke"}]
# Magic Mushrooms
pattern9 = [{"LOWER": "magic"}, {"LOWER": "mushrooms"}]
# wicked x
pattern10 = [{"LOWER": "wicked"}, {"LOWER": "x"}]
# sacred mush
pattern11 = [{"LOWER": "sacred"}, {"LOWER": "mush"}]
# lope dope
pattern12 = [{"LOWER": "lope"}, {"LOWER": "dope"}]


# create a Matcher
matcher = Matcher(nlp.vocab, validate=True)


# Add the pattern to the matcher
matcher.add("DRUG", None, pattern0, pattern1, pattern2, pattern3, pattern4, pattern5,
            pattern6, pattern7, pattern8, pattern9, pattern10, pattern11, pattern12)

#################################################
# Process texts and run the matcher
for doc in nlp.pipe(randomTexts):
    matches = matcher(doc)

    for match_id, start, end in matches:
        entType = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        print(entType, span.text)
##################################################


TRAINING_DATA = []

# Create a Doc object for each text in TEXTS
for doc in nlp.pipe(randomTexts):
    # Match on the doc and create a list of matched spans
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    # Get (start character, end character, label) tuples of matches
    entities = [(span.start_char, span.end_char, "DRUG") for span in spans]
    # Format the matches as a (doc.text, entities) tuple
    training_example = (doc.text, {"entities": entities})
    # Append the example to the training data
    TRAINING_DATA.append(training_example)

# print(*TRAINING_DATA, sep="\n")

# dump training data to JSON
with open('trn_data.json', 'w') as jsonf:
    json.dump(TRAINING_DATA, jsonf)
