#improved model
import pandas as pd
import re
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

df_train = pd.read_csv('trg.csv')
df_test = pd.read_csv('tst.csv')

df_train = df_train.rename(columns={'class': 'class_labels'})
train_labels = df_train['class_labels']

#text preprocessing
def text_processor(text):
    text = re.sub(r'\b(?<!-)\d+\b(?!-)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_inputs = df_train['abstract'].apply(text_processor)

test_inputs = df_test['abstract'].apply(text_processor)

# Oversampling the data
ros = RandomOverSampler(random_state=42)
train_inputs_resampled, train_labels_resampled = ros.fit_resample(np.array(train_inputs).reshape(-1, 1), train_labels)

# Flatten resampled data
train_inputs_resampled = train_inputs_resampled.flatten()

#spliting the data into train and validation set
X_train, X_val, Y_train, Y_val = train_test_split(train_inputs_resampled, train_labels_resampled, test_size=0.1, random_state=42)


vectorizer = CountVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(X_train)

X_val = vectorizer.transform(X_val)

X_test = vectorizer.transform(test_inputs)

class NaiveBayes:
    
    #calculates the prior_probability
    def prior_probability(self,X,Y):
        rows,features = X.shape
        unique_labels = list(Y.unique())
        prior = []
        
        for labels in unique_labels:
            X_labels = X[Y==labels]
            prior.append(X_labels.shape[0]/rows)
        return prior
        
    #calculates the conditional_probability
    def conditional_probability(self,X,Y):
        rows,features = X.shape
        unique_labels = list(Y.unique())
        conditional = []
        
        for labels in unique_labels:
            X_labels = X[Y==labels]
            frequency_sum = X_labels.sum(axis=0)
            words = X_labels.sum()
            conditional_prob = []
            
            for i in range(features):
                count = frequency_sum[0,i]
                #laplace smoothing
                prob = (count+1)/(words+features)
                conditional_prob.append(prob)
                
            conditional.append(conditional_prob)

        return conditional
        
    #makes a predction 
    def predict(self,X,Y,test):
        prior = self.prior_probability(X,Y)
        conditional = self.conditional_probability(X,Y)
        unique_labels = list(Y.unique())
        predictions = []
        
        for i in range(test.shape[0]):
            test_prob = []
            
            for j in range(len(prior)):
                prediction_prob = math.log(prior[j])
                
                for k in range(test.indptr[i], test.indptr[i+1]):
                    row = j
                    column = test.indices[k]
                    freq = test.data[k]
                    prediction_prob+=math.log(math.pow(conditional[row][column],freq))
                    
                test_prob.append(prediction_prob)
                
            prediction = unique_labels[test_prob.index(max(test_prob))]
            predictions.append(prediction)
                
        return predictions

bayes = NaiveBayes()
predictions = bayes.predict(X_train,Y_train,X_test)

#computes the accuracy 
# accuracy = accuracy_score(Y_val,predictions)
# print(accuracy)

counts = Counter(predictions)
print(counts)
df_predictions = pd.DataFrame({'id': df_test['id'], 'class': predictions})
df_predictions.to_csv('predictions.csv', index=False)
