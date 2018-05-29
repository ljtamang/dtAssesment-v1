
# coding: utf-8

# In[7]:


import gensim
import xml.etree.ElementTree as etree 
from random import shuffle
import multiprocessing
import numpy as np
from math import*
import csv
import utils


# In[8]:


# Get training data for doc2vec model
def get_all_paragraphs(instances):
    """Extract all sentences from instance and form single list of all contnet
    """
    ProblemDescription = []
    Question = []
    Answer = []
    ReferenceAnswers =[]
    for instance in instances:
        ProblemDescription.append(instance.ProblemDescription)
        Question.append(instance.Question)
        Answer.append(instance.Answer)
        ReferenceAnswers.extend(instance.ReferenceAnswers)
    all_paragraphs = ProblemDescription + Question + Answer + ReferenceAnswers
    return all_paragraphs

# prepare paragraph to be ready to feed to genism doc2vec model
def get_taggeDocument(paragraphs, tokens_only=False):
    """Converts the list of paragraphs to object of TaggedDocument to be ready to feed to doc2vec model
    """
    for i, paragraph in enumerate(paragraphs):
        if tokens_only:
            yield gensim.utils.simple_preprocess(paragraph)
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(paragraph), [i])


# In[4]:


# step to build the model

file_name ="DT-Gradev1.0_data/DT-Grade_v1.0_dataset.xml"
corpus = list(read_dtGrade_corpus(file_name)) # load dataset

# form training data
all_paragraphs = get_all_paragraphs(corpus)
shuffle(all_paragraphs)
training_data = list(get_taggeDocument(all_paragraphs))

# Build models and save them
cores = multiprocessing.cpu_count() # count number of processor 
model_DM = gensim.models.doc2vec.Doc2Vec(dbow_words=1, vector_size=300, window=8, min_count=1, sample=1e-5, negative=5, workers=cores,  dm=1, dm_concat=1, epochs=1000, alpha=0.025, min_alpha=0.0001)
model_DBOW = gensim.models.doc2vec.Doc2Vec(dbow_words=1, vector_size=300, window=5, min_count=1, sample=1e-5, negative=5, workers=cores, dm=0, dm_concat=1, epochs=400, alpha=0.025, min_alpha=0.0001)
print("Start buidling vocab for PV-DM model...")
model_DM.build_vocab(training_data) # build vocab
print("Finished buidling vocab for PV-DM model")

print("Start buidling vocab for PV-DBOW model....")
model_DBOW.build_vocab(training_data) #build vocab
print("Finished buidling vocab for PV-DM model")

print("Start buidling PV-DM model...")
get_ipython().run_line_magic('time', 'model_DM.train(training_data, total_examples=model_DM.corpus_count, epochs=model_DM.epochs)')
print("Finished buidlingPV-DM model")

print("Start buidling PV-DBOW model....")
model_DM.save('models/model_DM.doc2vec') #save model
get_ipython().run_line_magic('time', 'model_DBOW.train(training_data, total_examples=model_DBOW.corpus_count, epochs=model_DBOW.epochs)')
model_DBOW.save('models/model_DBOW.doc2vec') #save model
print("Finished buidling PV-DM model")

