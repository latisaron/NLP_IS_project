import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

# nltk.download("punkt")
# nltk.download("stopwords")
df = pd.read_csv(".//train.csv", header = 'infer', sep = ",", index_col = "id")

def tokenize(full_string, stop_words, stemmer):
    non_ascii_full_string = full_string.encode("ascii", "ignore").decode()
    tokenized_string = wordpunct_tokenize(non_ascii_full_string)
    tokens_without_stopwords = [word for word in tokenized_string if word not in stop_words]
    singles = [stemmer.stem(token) for token in tokens_without_stopwords]
    final_string = (' ').join(singles)
    return final_string

class_col = df.pop("label")
stop_words = stopwords.words("english")
stemmer = PorterStemmer()
df["tweet"] = [tokenize(row, stop_words, stemmer) for row in df["tweet"]]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["tweet"])
ch2 = SelectKBest(chi2, k = 50)
ch2.fit(x, class_col)
mask = ch2.get_support()
df["tweet"] = list(vectorizer.fit_transform(df["tweet"]).toarray())
df["tweet"] = [ row[mask] for row in df["tweet"] ]

column_names = ["feature" + str(i) for i in range(len(df["tweet"][1]))]
df_preprocessed = pd.DataFrame(df.tweet.to_list(), columns = column_names, index = df.index)

def metrics(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)    
    sensitivity = tp / (tp + fn)
    accuracy = (tp+tn) / (tp + fp + tn + fn)
    print("Accuracy has a value of {}".format(accuracy))
    print("Specificity has a value of {}".format(specificity))
    print("Sensitivity has a value of {}".format(sensitivity))
    print( pd.DataFrame(
        confusion_matrix(y_test, y_pred), 
        columns = ['Prediction 0', 'Prediction 1'],
        index = ['True 0', 'True 1']))

X_train, X_test, y_train, y_test = train_test_split(df_preprocessed, class_col, test_size=0.2, random_state=9, stratify=class_col)
svm = SVC(gamma = 'auto', probability = True, C = 10.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

metrics(y_test, y_pred)
