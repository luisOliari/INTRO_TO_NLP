from utils import db_connect
engine = db_connect()

# importo librerias
import pandas as pd
import pickle
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# leer el dataset:
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')

# vemos las primeras 5 columnas:
df_raw.head(5)

# infrmación del dataset:
df_raw.info()

df_raw['is_spam'].value_counts()

# hacemos una copia del dataset:
df = df_raw.copy()

# sacamos duplicados
df = df.drop_duplicates().reset_index(drop = True)

# funcón para eliminar https
def url(text):
    return re.sub(r'(https://www|https://)', '', text)

# se limpia url
df['url_limpia'] = df['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)

df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)

#Step 2: Usar técnicas de NLP para preprocesamiento de datos

vec = CountVectorizer().fit_transform(df['url_limpia'])

X_train, X_test, y_train, y_test = train_test_split(vec, df['is_spam'], stratify = df['is_spam'], random_state = 2207)

classifier = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))

# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score -> ",accuracy_score(predictions, y_test)*100)

# optimizo hiperparámetros
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(random_state=1234),param_grid,verbose=2)
grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_

predictions = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, predictions))








