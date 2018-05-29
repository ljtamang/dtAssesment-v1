
# coding: utf-8

# In[3]:


import gensim
import xml.etree.ElementTree as etree 
from random import shuffle
import numpy as np
from math import*
import csv


# In[5]:


def square_rooted(x):
    """Compute ssquare root
    """
    return round(sqrt(sum([a*a for a in x])),5)
    
def cosine_similarity(x,y):
    """Compute the cosine similarity score betwee two vectors
    Returns simialrity score
    """
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),5)
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

