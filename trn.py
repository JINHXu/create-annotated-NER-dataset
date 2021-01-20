'''
Author: Jinghua Xu
Hornor Code: I pledge that the code repreesent my own work.
Description: Read training data from the JSON fiile. Runs a training loop, and display the results with displacy.
'''

import spacy
import random
import json
import sys
from spacy import displacy

# Read training data from json
if len(sys.argv) < 2:
    sys.exit('Too few arguments, please speciify the input file')

filename = sys.argv[1]
# Load the reddit comments
with open(filename, 'r', encoding="utf-8") as f:
    TRAINING_DATA = json.loads(f.read())


# Setting up the pipeline
nlp = spacy.blank("en")

# Create a new entity recognizer and add it to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)

# Add the label "DRUG" to the entity recognizer
ner.add_label("DRUG")
# Start the training
nlp.begin_training()

# Loop for 10 iterations
for itn in range(10):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}

    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]

        # Update the model
        nlp.update(texts, annotations, losses=losses)
    print(losses)
    '''
    The losses printed on each iteration generally decrease as more iterations are performed as expected, one run:

    **kwargs
    {'ner': 1337.7450870524585}
    {'ner': 65.2017563708259}
    {'ner': 30.993893834290294}
    {'ner': 32.710884145085544}
    {'ner': 21.386925732606468}
    {'ner': 34.955316249868176}
    {'ner': 7.5803451807365985}
    {'ner': 1.9035998611860392}
    {'ner': 3.5095282618078785}
    {'ner': 8.200265490782776e-08}
    '''

# try the model on some (~10 - 20) hard-coded texts that are not part of the training data
# list of texts
txts = [
    "I think Rose is high on lsd.",
    "I think Lily is high on LSD.",
    "Jack takes cocaine.",
    "Andy overdosed heroin and he died.",
    "I ate too much food, I feel sick.",
    "Andy has been staying away from alcohol since he had a stroke.",
    "We all know that marijuana can get you high.",
    "I don't do drugs.",
    "He screwed up coz he was high on PCP.",
    "Ketamine is a medication primarily used for starting and maintaining anesthesia.",
    "My friend Anderson has stopped taking keratom a long time ago.",
    "Nicotine can kill you",
    "We all know nicotine can kill you.",
    "Hugh took some ketamine last night.",
    "Was he high on pcp?",
    "She took too much keratom.",
    "I need some mdma.",
    "I need some MDMA.",
    "I used some laughing gas to make myself feel better.",
    "I am not s big fan of lope dope."
]

docs = []

for txt in txts:
    doc = nlp(txt)
    [print(ent.label_, ent.text) for ent in doc.ents if ent.label_ in ['DRUG']]

    hasDRUG = False
    for ent in doc.ents:
        if ent.label_ == 'DRUG':
            hasDRUG = True
            break

    if hasDRUG:
        docs.append(doc)

displacy.serve(docs, style="ent", options={"ents": ["DRUG"]})
displacy.serve(docs, style="dep")
