
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[7]:



# load dataset into Pandas DataFrame
data_path = 'output/output_features.csv' # define path of data
dataFrame = pd.read_csv(data_path, header=None)

# Define features and target
header = list(dataFrame.columns.values)
features = header[:len(header)-1]
target = header[len(header)-1]

# Standardize the features
x = dataFrame.loc[:, features].values # Separate out the feature
y = dataFrame.loc[:,[target]] # Separate out the target
x_standarized = StandardScaler().fit_transform(x)

# PCA projection to reduce dimension
pca = PCA(.98) # choose the minimum number of principal components such that 95% of the variance is retained
principal_components = pca.fit_transform(x_standarized)
principal_df = pd.DataFrame(data = principal_components) # convert np.ndarray to dataframe
final_df = pd.concat([principal_df, dataFrame[[target]]], axis = 1)

# save final reduced dataframe to csv
output_path = 'output/pca_features.csv' # define path of data
final_df.to_csv(output_path, header=False, index=False)


# In[50]:


# load the data after converting class lablel in pca_features.csv to number(nominal) using excel
data_path = 'output/pca_features.csv' # define path of data
df = pd.read_csv(data_path, header=None)


# In[51]:


from imblearn.over_sampling import RandomOverSampler

header = list(df.columns.values)
features = header[:len(header)-1]
target = header[len(header)-1]

# upsample using Naive random over-sampling
# detail : http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html
x = df.loc[:, features].values # Separate out the feature
y = df.loc[:,[target]] # Separate out the target
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(x, y.values.ravel())
X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)
final_df = pd.concat([X_resampled, y_resampled], axis = 1)                    

# save to csv file
output_path = 'output/pca_features_upsampled.csv' # define path of data
final_df.to_csv(output_path, header=False, index=False)

