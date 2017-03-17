
# coding: utf-8

# In[232]:

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score


import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[117]:

imdb_data = pd.read_csv('movie_metadata.csv')


# In[118]:

imdb_data.head(5)


# # Cleaning the data

# Removing all non numeric data.

# In[119]:

str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in imdb_data.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = imdb_data.columns.difference(str_list)


# I tried filling values for this label with Zeros and the mean of the values and I the mean worked better. It resulted in more precision and recall

# In[120]:

imdb_data['imdb_score']=imdb_data['imdb_score'].fillna(np.mean).astype(int)



# In[346]:

movie_num = imdb_data[num_list]


# In[348]:

movie_num.head(5)


# In[122]:

movie_num = movie_num.fillna(value = 0, axis = 1)


# Specifying my features. I picked all numeric data 15 features and my label will be the imdb_score. 

# In[123]:

movie_feat = movie_num.drop('imdb_score', axis = 1 )


# Normalizing the data taking the variance of the data into consideration and neglecting the mean. the standard scaler works better than normalizing the data to a unit norm 

# In[124]:

scaler = StandardScaler()
mov_scaled = scaler.fit_transform(movie_feat)


# In[289]:

movie_num['imdb_score'].value_counts()


# In[125]:

z = movie_num['imdb_score'].values


# In[177]:

z


# I'm using the train test split method. Using 70% of the data for training and 30% for testing. It's random, effecient and serves as a check for overfitting.
# 
# Split the dataset into two pieces, so that the model can be trained and tested on different data.
# 
# Better estimate of out-of-sample performance, but still a "high variance" estimate

# In[127]:

X_train, X_test, y_train, y_test = train_test_split(mov_scaled, z, test_size=0.3, random_state=0)


# # The classifier

# The data is not linearly seperable, so Linear discriminant was pretty bad. I tried first naive bayes, but since the data is not normally distributed I was getting bad results, KNN also was bad, because it was neglecting the distribution of the classes. Then Decision trees didn't provide that much improvement, so  I used ensembling and boosting techniques to improve my results. Random forest and Gradient boosting  raised the accuracy to over 50%. The no.of estimators parameter improved the accuracy a lot. 

# Ensemble model combines multiple individual models together to enhance prediction. 
# There is low correlation between the features, so ensemble in general  should work fine with that(Random Forest). The forest basically chooses the classification having the most votes (over all the trees in the forest). 

# In[274]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf=RandomForestClassifier(n_estimators=50, max_depth=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
"RandomForest",accuracy_score(y_test,y_pred)


# # Measuring performance 

# I got an accuracy of almost 53% coorectly classified. 

# the confusion matrix tells me all about accuracy, precision and recall 

# In[164]:

conf = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
print(prec)
recall = recall_score(y_test, y_pred, average='weighted')
print(recall)


# In[316]:

conf


# In[334]:

precision5 = conf[4,4]/sum(conf[:,4])
print(precision5)
recall5 = conf[4,4]/sum(conf[4])
print(recall5)


# In[335]:

precision6 = conf[5,5]/sum(conf[:,5])
print(precision7)
recall6 = conf[5,5]/sum(conf[5])
print(recall6)


# In[336]:

precision7 = conf[6,6]/sum(conf[:,6])
print(precision7)
recall7 = conf[6,6]/sum(conf[6])
print(recall7)


# Because there isn't enough examples for some classes the model isn't able to generalize very well. but it 's getting a precision of around 52% 
# 
# which means when a true positive label is predicted the model is 52% likely to predict it correctly. 
# 
# 
# it measures how precise our model is when predicting a specific label !
# 
# 
# 

# Recall means when the actual label is predicted. how often is the prediction correct !
# 
# It's the rate of the true positive values 

# Again the inconsistency of the data plays the biggest part we don't have enough data for some labels so the model is not able to generalize pretty well

# In[345]:

plt.figure(figsize = (12,10))
plt.title('Confusion Matrix ')
sns.heatmap(conf, annot=True)


# Boosting 
# boosting is an iterative technique it adjusts the weight of an observation based on the last classification. I was trying to make sure the model is not Biased and that it's not overfitting. I used a learning rate of 0.1 to avoid being stuck in local minimum.

# In[279]:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
clf=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
clf.fit(X_train,y_train)
y_pred1 = clf.predict(X_test)
print("Gradient Boosting",accuracy_score(y_test,y_pred1))
prec1 = precision_score(y_test, y_pred1, average='weighted')
print(prec1)
recall1 = recall_score(y_test, y_pred1, average='weighted')
print(recall1)


# # Trying to predict the score using the likes on facebook 

# In[29]:

my=list(zip(imdb_data['director_facebook_likes'],imdb_data['actor_1_facebook_likes'],imdb_data['actor_2_facebook_likes'],imdb_data['actor_3_facebook_likes']))
X = np.array(my)

y = z
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[31]:

imdb_data['num_critic_for_reviews']=imdb_data['num_critic_for_reviews'].fillna(0.0).astype(np.float32)
imdb_data['director_facebook_likes']=imdb_data['director_facebook_likes'].fillna(0.0).astype(np.float32)
imdb_data['actor_3_facebook_likes'] = imdb_data['actor_3_facebook_likes'].fillna(0.0).astype(np.float32)
imdb_data['actor_1_facebook_likes'] = imdb_data['actor_1_facebook_likes'].fillna(0.0).astype(np.float32)
imdb_data['gross'] = imdb_data['gross'].fillna(0.0).astype(np.float32)
imdb_data['num_voted_users'] = imdb_data['num_voted_users'].fillna(0.0).astype(np.float32)
imdb_data['cast_total_facebook_likes'] = imdb_data['cast_total_facebook_likes'].fillna(0.0).astype(np.float32)
imdb_data['num_user_for_reviews'] = imdb_data['num_user_for_reviews'].fillna(0.0).astype(np.float32)
imdb_data['facenumber_in_poster'] = imdb_data['facenumber_in_poster'].fillna(0.0).astype(np.float32)
imdb_data['actor_2_facebook_likes'] = imdb_data['actor_2_facebook_likes'].fillna(0.0).astype(np.float32)
imdb_data['budget'] = imdb_data['budget'].fillna(0.0).astype(np.float32)
imdb_data['movie_facebook_likes'] = imdb_data['movie_facebook_likes'].fillna(0.0).astype(np.float32)


imdb_data.info()


# In[342]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
"RandomForest",accuracy_score(y_test,y_pred2)


# In[343]:

conf = confusion_matrix(y_test, y_pred2)
prec = precision_score(y_test, y_pred2, average='weighted')
print(prec)
recall = recall_score(y_test, y_pred2, average='weighted')
print(recall)


# Using social media numbers only. it produced numbers close to previous numbers 

# In[ ]:



