
# coding: utf-8

# In[6]:


import gensim
import xml.etree.ElementTree as etree 
from random import shuffle
import multiprocessing
import numpy as np
from math import*
import csv


# In[7]:


def get_annotation(instance):
    """Get the annotation.
    Given annotation in the form: <Annotation Label="correct(1)|correct_but_incomplete(0)|contradictory(0)|incorrect(0)">
    It returns the one of corresonding label which has 1 ie. correct in the above given annotation
    """
    annotation_label = instance[4].attrib['Label']
    annoted_position = [int(s) for s in annotation_label if s.isdigit()].index(1)
    if annoted_position is 0:
        return "correct"
    elif annoted_position is 1:
        return "correct_but_incomplete"
    elif annoted_position is 2:
        return "contradictory"
    elif annoted_position is 3:
        return "incorrect"
    
def get_refrence_answer(instance):
    """Parse the refrence answer.
    Returns the list of refrence asnwers
    """
    ref_answers = instance[5].text
    ref_answers = ref_answers.splitlines()
    answers = [answer.split(":", 1)[1].strip() for answer in ref_answers[1:]]
    return answers
class Instance(object):
    def __init__(self, instance):
        self.id = instance.attrib['ID']
        self.ProblemDescription = instance[1].text
        self.Question = instance[2].text
        self.Answer = instance[3].text
        self.Annotation = get_annotation(instance)
        self.ReferenceAnswers = get_refrence_answer(instance) 


# In[8]:


#read DT-Gradev1 corpus
def read_dtGrade_corpus(file_name):
    """Read the dtGrade  corpus 
    Returns list of object of Instance type
    """
    tree = etree.parse(file_name) # load entire document as an object
    root = tree.getroot() # get refrence to the root element i.e. instances
    instances = root.findall('{http://www.w3.org/2005/Atom}Instance')
    for instance in instances:
        yield Instance(instance)   


# In[9]:


# load dataset
file_name ="DT-Gradev1.0_data/DT-Grade_v1.0_dataset.xml"
corpus = list(read_dtGrade_corpus(file_name)) #returns list of object of Instance type and shuffle them


# In[10]:


# load back the model 
model_DM = gensim.models.doc2vec.Doc2Vec.load('models/model_DM.doc2vec')
# model_DBOW = gensim.models.doc2vec.Doc2Vec.load('models/model_DBOW.doc2vec')


# In[59]:


# create file with feature for answer, refrence answer and answer annotation
with open('output/output_features.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)

    for instance in corpus:
        # infer vector for student answer
        vec_DM_ans = model_DM.infer_vector(gensim.utils.simple_preprocess(instance.Answer))

        #infer vector for refrence answer
        vec_DM_refAns = []
        for refAns in instance.ReferenceAnswers:
            vec_refAns = model_DM.infer_vector(gensim.utils.simple_preprocess(refAns))
            vec_DM_refAns.append(vec_refAns)   
        vec_DM_refAns_mean = np.mean(np.array(vec_DM_refAns), axis=0) # mean of all refrence answer's vector

        # form row to print
        row = vec_DM_ans.tolist()+vec_DM_refAns_mean.tolist()+[instance.Annotation]
        spamwriter.writerow(row)
        

