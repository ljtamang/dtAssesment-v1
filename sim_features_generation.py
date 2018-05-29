
# coding: utf-8

# In[13]:


import gensim
import xml.etree.ElementTree as etree 
from random import shuffle
from math import*
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils


# In[10]:


# load back the model 
model_DM = gensim.models.doc2vec.Doc2Vec.load('models/model_DM.doc2vec')
model_DBOW = gensim.models.doc2vec.Doc2Vec.load('models/model_DBOW.doc2vec')

# load dataset
file_name ="DT-Gradev1.0_data/DT-Grade_v1.0_dataset.xml"
corpus = list(read_dtGrade_corpus(file_name)) #returns list of object of Instance type and shuffle them

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
with open('sim_features/similarity.csv', "w") as csv_file:
    fieldnames = ['DM_max_cosSim', 'DM_mean_cosSim', 'DBOW_max_cosSim', 'DBOW_mean_cosSim' ,'DM_DBOW_max_cosSim', 'DM_DBOW_mean_cosSim', 'Annotation']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for instance in corpus:
        DM_cosSim, DBOW_cosSim, DM_DBOW_cosSim = get_cosim_score(instance.Answer, instance.ReferenceAnswers)
        writer.writerow({'DM_max_cosSim': max(DM_cosSim), 'DM_mean_cosSim': round(np.mean(DM_cosSim),5), 'DBOW_max_cosSim': max(DBOW_cosSim), 'DBOW_mean_cosSim': round(np.mean(DBOW_cosSim),5), 'DM_DBOW_max_cosSim': max(DM_DBOW_cosSim), 'DM_DBOW_mean_cosSim': round(np.mean(DM_DBOW_cosSim),5), 'Annotation':instance.Annotation})


# In[11]:


# load dataset into Pandas DataFrame
data_path = 'sim_features/similarity.csv' # define path of data
dataFrame = pd.read_csv(data_path)

# Define features and target
features = ['DM_max_cosSim', 'DM_mean_cosSim', 'DBOW_max_cosSim', 'DBOW_mean_cosSim', 'DM_DBOW_max_cosSim', 
            'DM_DBOW_mean_cosSim']

# Standardize the features
x = dataFrame.loc[:, features].values # Separate out the feature
y = dataFrame.loc[:,['Annotation']] # Separate out the target

x_standarized = StandardScaler().fit_transform(x)

# save the standarized data to csv
x = pd.DataFrame(x_standarized)
y = pd.DataFrame(y)
final_df = pd.concat([x, y], axis = 1) 
header_names= features + ['Annotation']
output_path = 'sim_features/similarity_standarised.csv' # define path of data
final_df.to_csv(output_path, header=header_names, index=False)


# In[12]:


from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle

data_path = 'sim_features/similarity_standarised.csv' # define path of data
dataFrame = pd.read_csv(data_path)

# Define features and target
features = ['DM_max_cosSim', 'DM_mean_cosSim', 'DBOW_max_cosSim', 'DBOW_mean_cosSim', 'DM_DBOW_max_cosSim', 
            'DM_DBOW_mean_cosSim']

# Standardize the features
x = dataFrame.loc[:, features].values # Separate out the feature
y = dataFrame.loc[:,['Annotation']] # Separate out the target

# upsample using Naive random over-sampling
# detail : http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(x, y.values.ravel())
X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)
final_df = pd.concat([X_resampled, y_resampled], axis = 1)                    

# save to csv file
output_path = 'sim_features/similarity_standarised_overSampled.csv' # define path of data
final_df = shuffle(final_df)
header_names= features + ['Annotation']
final_df.to_csv(output_path,header=header_names, index=False)

