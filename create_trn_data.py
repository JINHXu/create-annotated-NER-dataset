'''
Author: Jinghua Xu
Hornor Code: I pledge that the code repreesent my own work.
Description: Create training data for NER from reddit posts/comments using patterns and SpaCy. Save training data to a file.
'''
import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
import json
import sys
import random

############################################
# Token-based Matching
# https://spacy.io/usage/rule-based-matching
############################################

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

######################################################
# patterns to match more patterns tba
######################################################

# link to commonly used drugs: https://www.drugabuse.gov/sites/default/files/Commonly-Used-Drugs-Charts_final_June_2020_optimized.pdf

# Patterns
ingredient_pattern = {
    'label': 'DRUG',
    'pattern': [
        {
            'LEMMA': { 'DR': ['alcohol', 'ayahuasca', 'cocaine', 'Dimethyltriptamine', 'DMT', 'Gamma-hydroxybutyrate', 'GHB', 'hallucinogens', 'Heroin', 'ketamine', 'khat', 'kratom', 'LSD', 'Marijuana', 'MDMA', 'peyote', 'methamphetamine', 'dextromethorphan', 'DXM', 'loperamide', 'PCP', 'opioids', 'psilocybin', 'flunitrazepam', 'rohypnol', 'salvia', 'anabolic', 'steroids', 'tabacco', 'nicotine'] },
            'POS': 'NOUN'
        }
    ]
}

patterns = [ingredient_pattern]

# Create an Entity Ruler and add patterns
ruler = EntityRuler(nlp, overwrite_ents=True)
ruler.add_patterns(patterns)

# Add the Entity Ruler to the nlp pipeline
nlp.add_pipe(ruler, after="ner")

# Process texts with the Entity Ruler in the pipelne
# Process the texts one at a time because if nlp.pipe(randomTexts) is used,
# displacy doesn't work

docs = []
for text in randomTexts:
    doc = nlp(text)
    [print(ent.label_, ent.text) for ent in doc.ents if ent.label_ in ['DRUG']]

    hasING = False
    for ent in doc.ents:
        if ent.label_ == 'DRUG':
            hasING = True
            break

    if hasING:
        docs.append(doc)
'''
# a pattern to match common drug names
pattern1 = [
    {
        'LEMMA': {'DRU': ['alcohol']},
        # , 'ayahuasca', 'cocaine', 'Dimethyltriptamine', 'DMT', 'Gamma-hydroxybutyrate', 'GHB', 'hallucinogens', 'Heroin', 'ketamine', 'khat', 'kratom', 'LSD', 'Marijuana', 'MDMA', 'peyote', 'methamphetamine', 'dextromethorphan', 'DXM', 'loperamide', 'PCP', 'opioids', 'psilocybin', 'flunitrazepam', 'rohypnol', 'salvia', 'anabolic', 'steroids', 'tabacco', 'nicotine']},
        'POS': 'NOUN'
    }
]

# create a Matcher
matcher = Matcher(nlp.vocab, validate = True)


# Add the pattern to the matcher
matcher.add("DRUG", None, pattern1)


# Process texts and run the matcher
for doc in nlp.pipe(randomTexts):
    matches = matcher(doc)

    for match_id, start, end in matches:
        entType = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        print(entType, span.text)
'''

'''
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

print(*TRAINING_DATA, sep="\n")


# dump to JSON

'''