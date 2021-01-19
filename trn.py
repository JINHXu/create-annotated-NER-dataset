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

# try the model on some (~10 - 20) hard-coded texts that are not part of the training data,
# but that you think the model should find. Display the results with displacy.

'''
[
    ["How to preorder the iPhone X", { "entities": [[20, 28, "GADGET"]] }],
    ["iPhone X is coming", { "entities": [[0, 8, "GADGET"]] }],
    ["Should I pay $1,000 for the iPhone X?", { "entities": [[28, 36, "GADGET"]] }],
    ["The iPhone 8 reviews are here", { "entities": [[4, 12, "GADGET"]] }],
    ["Your iPhone goes up to 11 today", { "entities": [[5, 11, "GADGET"]] }],
    ["I need a new phone! Any tips?", { "entities": [] }]
]
'''

'''
txt = [
    ["I think Lily is high on LSD.", {"entities": [[24, 27, "DRUG"]]}],
    # drug names in upper or lower case should all be recognized
    ["I think Rose is high on lsd.", {"entities": [[24, 27, "DRUG"]]}],
    ["Jack takes cocaine.", {[[11, 18, "DRUG"]]}],
    ["Andy overdosed heroin and he died.", {[[15, 21, "DRUG"]]}],
    # gamma-hydroxybutyrate
    ["I ate too much food, I feel sick.", {[[]]}]
    # more items go here
]
'''

txt = [
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
    "I need some MDMA."
]

'''
doc1 = nlp("I think Lily is high on heroin.")
print("Entities", [(ent.text, ent.label_) for ent in doc1.ents])

doc2 = nlp("Jack overdosed LSD.")
print("Entities", [(ent.text, ent.label_) for ent in doc2.ents])
'''

docs = nlp.pipe(txt)
nlp.add_pipe(nlp.create_pipe('sentencizer'))

sents = []

for doc in docs:
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    # Collect the sentences in a list, for deps visualization
    for sent in doc.sents:
        sents.append(sent)
        #######################
        print(sent)

# so far I would not say the model does very well on recognizing, mostly only 5 or 6 can be recognized

# view named entities
# Display named entities
#
# displacy.serve(docs, style="ent")

# print(spacy.displacy.render(docs, style="ent", page="true"))

# view dependency trees

# Display dependency parses for all sentences in the texts, including lemmas
displacy.serve(sents, style="dep", options={"add_lemma": True})
