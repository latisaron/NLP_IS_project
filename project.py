import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

def tokenize(full_string, stop_words, normalizer):
    ascii_full_string = full_string.encode("ascii", "ignore").decode()
    tokenized_string = wordpunct_tokenize(ascii_full_string)
    tokens_without_stopwords = [word for word in tokenized_string if word not in stop_words]
    if type(normalizer) == nltk.stem.porter.PorterStemmer:
        singles = [normalizer.stem(token) for token in tokens_without_stopwords]
    if type(normalizer) == nltk.stem.wordnet.WordNetLemmatizer:
        singles = [normalizer.lemmatize(token) for token in tokens_without_stopwords]
    final_string = (' ').join(singles)
    return final_string

def metrics(y_test, y_pred, alg):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)    
    sensitivity = tp / (tp + fn)
    accuracy = (tp+tn) / (tp + fp + tn + fn)
    print("============== {} ==============".format(alg))
    print("Accuracy has a value of {}".format(accuracy))
    print("Specificity has a value of {}".format(specificity))
    print("Sensitivity has a value of {}".format(sensitivity))
    print( pd.DataFrame(
        confusion_matrix(y_test, y_pred), 
        columns = ['Prediction 0', 'Prediction 1'],
        index = ['True 0', 'True 1']))
    print(" ")
    
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download('wordnet')
df = pd.read_csv(".//train.csv", header = 'infer', sep = ",", index_col = "id")
class_1_df = df[df["label"] == 1]
class_0_df = df[df["label"] == 0].sample(frac = 1)[ : 2242]
df = pd.concat([class_0_df, class_1_df]).sample(frac = 1)

class_col = df.pop("label")
stop_words = stopwords.words("english")
normalizer_hash = {
    "stemmer": PorterStemmer(),
    "lemmatizer": WordNetLemmatizer()
    }
df["tweet"] = [tokenize(row, stop_words, normalizer_hash["lemmatizer"]) for row in df["tweet"]]

vectorizer_hash = {
    "tfidf": TfidfVectorizer(),
    "count": CountVectorizer()
    }
x = vectorizer_hash["count"].fit_transform(df["tweet"])
ch2 = SelectKBest(chi2, k = 50)
ch2.fit(x, class_col)
mask = ch2.get_support()
df["tweet"] = list(x.toarray())
df["tweet"] = [ row[mask] for row in df["tweet"] ]

column_names = ["feature" + str(i) for i in range(len(df["tweet"].iloc[1]))]
df_preprocessed = pd.DataFrame(df.tweet.to_list(), columns = column_names, index = df.index)


X_train, X_test, y_train, y_test = train_test_split(df_preprocessed, class_col, test_size=0.2, random_state=9, stratify=class_col)
svm = SVC(gamma = 'auto', probability = True, C = 10.0)
parameters = {
            'C': [5, 10, 15]
}
# clf = GridSearchCV(svm, parameters, cv=5)
# clf.fit(X_train, y_train)
# y_pred_svm = clf.predict(X_test)

# metrics(y_test, y_pred_svm, "svm")

# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred_nb = gnb.predict(X_test)

# metrics(y_test, y_pred_nb, "naive bayes")


from torch.nn import ReLU
from torch.nn import Module
from torch.nn import Sigmoid
from torch.nn import Linear
from torch import Tensor
from torch.utils.data import *
from torch import LongTensor
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import xavier_uniform_
import numpy as np


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.hidden_layer1 = Linear(50, 25)
        xavier_uniform_(self.hidden_layer1.weight)
        self.act1 = ReLU()
        self.hidden_layer2 = Linear(25, 25)
        xavier_uniform_(self.hidden_layer2.weight)
        self.act2 = ReLU()
        self.hidden_layer3 = Linear(25, 10)
        xavier_uniform_(self.hidden_layer3.weight)
        self.act3 = ReLU()
        self.hidden_layer4 = Linear(10, 1)
        xavier_uniform_(self.hidden_layer3.weight)
        self.act4 = Sigmoid()
        
    def forward(self, data):
        data = self.hidden_layer1(data)
        data = self.act1(data)
        data = self.hidden_layer2(data)
        data = self.act2(data)
        data = self.hidden_layer3(data)
        data = self.act3(data)
        data = self.hidden_layer4(data)
        data = self.act4(data)
        return data

train = TensorDataset(Tensor(np.array(X_train)), LongTensor(np.array(y_train)))
train_dl = DataLoader(train, batch_size = 16)

mlp = MLP()    
optimizer = SGD(mlp.parameters(), lr = 1e-3)
criterion = BCELoss()

# training
for epoch in range(100):
    for i, data in enumerate(train_dl, 0):
        inputs, targets = data
        optimizer.zero_grad()
        y_pred = mlp(inputs)
        targets = targets.unsqueeze(1)
        targets = targets.float()
        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

# ugly testing
y_pred_mlp = []
for _, item in X_test.iterrows():
    if type(item) == str:
        continue
    y_pred_mlp.append(np.round(mlp(Tensor(np.array(item)))[0].item()))

metrics(y_test, y_pred_mlp, "mlp")
