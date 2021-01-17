'''
Author: Jinghua Xu
Hornor Code: I pledge that the code repreesent my own work.
Description: Read training data from the JSON fiile. Runs a training loop, and display the results with displacy.
'''
import spacy
import random
import json
import sys

# Read training data from json
if len(sys.argv) < 2:
    sys.exit('Too few arguments, please speciify the input file')

filename = sys.argv[1]
# Load the reddit comments
with open(filename, 'r', encoding="utf-8") as f:
    TRAINING_DATA = json.loads(f.read())


# Setting up the pipeline
# Create a blank "en" model
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
    {'ner': 6543.247388635707}
    {'ner': 1692.5676541115831}
    {'ner': 1080.256173849884}
    {'ner': 754.6397855908697}
    {'ner': 593.5560192089799}
    {'ner': 483.8135037326986}
    {'ner': 384.3991251286233}
    {'ner': 373.49454041613825}
    {'ner': 315.75702013047595}
    {'ner': 286.361619557513}
    '''

# try the model on some (~10 - 20) hard-coded texts that are not part of the training data,
# but that you think the model should find. Display the results with displacy.
