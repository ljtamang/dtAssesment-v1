
# coding: utf-8

import gensim
import xml.etree.ElementTree as etree 
from random import shuffle
import multiprocessing
import numpy as np
from math import*
import csv

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


# load dataset
file_name ="DT-Gradev1.0_data/DT-Grade_v1.0_dataset.xml"
corpus = list(read_dtGrade_corpus(file_name)) #returns list of object of Instance type and shuffle them

# form training data
all_paragraphs = get_all_paragraphs(corpus)
shuffle(all_paragraphs)
training_data = list(get_taggeDocument(all_paragraphs))


# Build models and save them
cores = multiprocessing.cpu_count() # count number of processor 
model_DM = gensim.models.doc2vec.Doc2Vec(dbow_words=1, vector_size=300, window=8, min_count=1, sample=1e-5, negative=5, workers=cores,  dm=1, dm_concat=1, epochs=1000, alpha=0.025, min_alpha=0.0001)
model_DBOW = gensim.models.doc2vec.Doc2Vec(dbow_words=1, vector_size=300, window=5, min_count=1, sample=1e-5, negative=5, workers=cores, dm=0, dm_concat=1, epochs=400, alpha=0.025, min_alpha=0.0001)
model_DM.build_vocab(training_data) # build vocab
model_DBOW.build_vocab(training_data) #build vocab
get_ipython().magic('time model_DM.train(training_data, total_examples=model_DM.corpus_count, epochs=model_DM.epochs)')
model_DM.save('models/model_DM.doc2vec') #save model
get_ipython().magic('time model_DBOW.train(training_data, total_examples=model_DBOW.corpus_count, epochs=model_DBOW.epochs)')
model_DBOW.save('models/model_DBOW.doc2vec') #save model
print("finished building model")


# load back the model 
model_DM = gensim.models.doc2vec.Doc2Vec.load('models/model_DM.doc2vec')
model_DBOW = gensim.models.doc2vec.Doc2Vec.load('models/model_DBOW.doc2vec')


# cacluate the simialrty score btween answer and refrence asnwer
def get_cosim_score(Answer, ReferenceAnswers):
    """Calcualte simialrity score between answer and refrence answers
    """
    DM_cosSim, DBOW_cosSim, DM_DBOW_cosSim= [],[],[]
    # infer answers vector
    v1_DM_ans = model_DM.infer_vector(gensim.utils.simple_preprocess(Answer))
    v2_DBOW_ans = model_DBOW.infer_vector(gensim.utils.simple_preprocess(Answer))
    # produce concatenation vector
    v1_v2_ans = np.concatenate([v1_DM_ans, v2_DBOW_ans]) 
    for refAns in ReferenceAnswers:
        refAns = gensim.utils.simple_preprocess(refAns)
        # infer refrence answers vector
        v1_DM_refAns = model_DM.infer_vector(refAns)
        v2_DBOW_refAns = model_DBOW.infer_vector(refAns)
        # produce concatenation vector
        v1_v2_refAns = np.concatenate([v1_DM_refAns, v2_DBOW_refAns])
        # find cosine simialities
        DM_cosSim.append(cosine_similarity(v1_DM_ans, v1_DM_refAns))
        DBOW_cosSim.append(cosine_similarity(v2_DBOW_ans, v2_DBOW_refAns))
        DM_DBOW_cosSim.append(cosine_similarity(v1_v2_ans, v1_v2_refAns))
    return DM_cosSim, DBOW_cosSim, DM_DBOW_cosSim


# compute simialrity score and write them to file
with open('output/scores.csv', "w") as csv_file:
    fieldnames = ['DM_max_cosSim', 'DM_mean_cosSim', 'DBOW_max_cosSim', 'DBOW_mean_cosSim' ,'DM_DBOW_max_cosSim', 'DM_DBOW_mean_cosSim', 'Annotation']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for instance in corpus:
        DM_cosSim, DBOW_cosSim, DM_DBOW_cosSim = get_cosim_score(instance.Answer, instance.ReferenceAnswers)
        writer.writerow({'DM_max_cosSim': max(DM_cosSim), 'DM_mean_cosSim': round(np.mean(DM_cosSim),5), 'DBOW_max_cosSim': max(DBOW_cosSim), 'DBOW_mean_cosSim': round(np.mean(DBOW_cosSim),5), 'DM_DBOW_max_cosSim': max(DM_DBOW_cosSim), 'DM_DBOW_mean_cosSim': round(np.mean(DM_DBOW_cosSim),5), 'Annotation':instance.Annotation})

