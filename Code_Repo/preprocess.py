import pandas as pd
import json
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics


"""
This test has been performed w/o doing any modifications in the training and test data
Overall accuracy is about 70%.. There are many False Positives as well
We need to think more how to increase the model accurracy after revisting the data once more
"""


"""
Read the Training and Test Data of jsonl format to pandas dataframe
Read training data
"""

def read_jsonl_to_dataFrame(filepath,dfColname1,dfColname2,dfColname3):
    new_list = []
    with open(filepath, 'r') as json_file:
        #with open('./data/train.jsonl', 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        new_list.append(json.loads(json_str))
    df = pd.DataFrame(new_list,columns=(dfColname1,dfColname2,dfColname3))
    return df
    
    

df_train = read_jsonl_to_dataFrame('../data/train.jsonl',"label","response","context")
print("Training data DataFrame -->",df_train.head())
df_test = read_jsonl_to_dataFrame('../data/test.jsonl',"id","response","context")
print("Test data DataFrame -->",df_test.head())


#print(df_train.isnull().sum())  
#print(df_test.isnull().sum())          

X = df_train['response']
y = df_train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Naïve Bayes:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])

# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])


# Fit the Naive Baise Model

text_clf_nb.fit(X_train, y_train)   

# Form a prediction set
predictions = text_clf_nb.predict(X_test)   

# Report the confusion matrix

print("Confusion Matrix Result for Naive Baise", metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print("Classification Result for Naive Baise", metrics.classification_report(y_test,predictions))   


# Print the overall accuracy
print("Overall accuracy for Naive Baise",metrics.accuracy_score(y_test,predictions))

# Fit the Linear Support Vector Classifier Model
text_clf_lsvc.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_lsvc.predict(X_test)


# Report the confusion matrix

print("Confusion Matrix Result for Linear SVC", metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print("Classification Result for Linear SVC", metrics.classification_report(y_test,predictions))   


# Print the overall accuracy
print("Overall accuracy for Linear SVC",metrics.accuracy_score(y_test,predictions))
    


    