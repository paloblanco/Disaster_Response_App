# import packages
import sys
from typing import Tuple, List

# data loading and export
import pandas as pd
import sqlite3
import pickle

# ML
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords','omw-1.4'])
from nltk.corpus import stopwords

import re
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

# Globals required for ML

# this is defined globally so it does not need to be compiled repeatedly in a function
PATTERN_REMOVE_STOPWORDS = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')


def load_data(messages_file: str, categories_file: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Reads in raw messages data, stores single table in sqlite, returns X and Y for model training.
    
    INPUTS:
        messages_file: str - csv file name containing messages
        categories_file: str - csv file containing categories for above messages.

    OUTPUTS:
        X: pd.Series - series of messages, dtype = text
        Y: pd.DataFrame - dataframe of all category labels corresponding to X

    -- A file is created as a result of this process:
        messages.sqlite - sqlite database
    """
    # read in file
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)
    df = messages.merge(categories,on="id")

    # clean data
    categories = df.categories.str.split(";",expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    categories = categories.replace(to_replace=2,value=1) # only 1s and 0s are allowed

    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df=df.drop_duplicates()

    # load to database
    dbname = "messages.sqlite"
    cnxn = sqlite3.connect(dbname)
    df.to_sql("messages_classified",cnxn,index=False, if_exists='replace')
    cnxn.close()

    # define features and label arrays
    X = df.message
    Y = df.iloc[:,4:]

    return X, Y

def tokenize(text: str) -> List[str]:
    """
    Tokenizer for messages. Returns a list of relevant words. 
    Intended for usage in a sklearn pipeline.

    INPUTS:
        text: str - string corresponding to a message

    OUTPUTS:
        clean_tokens: List[str] - list of tokens (relevant words) in supplied text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = PATTERN_REMOVE_STOPWORDS.sub('',text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in word_tokenize(text):
        clean_tok = lemmatizer.lemmatize(tok)#.lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> GridSearchCV:
    """
    Build a grid search object to be trained.

    INPUTS:
        None

    OUTPUTS:
        model_pipeling: GridSearchCV - grid search object which has "fit" and "predict" methods
    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ], verbose=True)

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [10, 20] # 40],
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def train(X, Y, model):
    """
    Train a model based on provided dataset

    INPUTS:
        X: pd.Series - series of messages, dtype = text
        Y: pd.DataFrame - dataframe of all category labels corresponding to X
        model:  GridSearchCV - grid search object which has "fit" and "predict" methods

    OUPUTS:
        model:  GridSearchCV - same object as input, now trained
    """
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # fit model
    model.fit(X_train, Y_train)

    # output model test results
    Y_pred = model.predict(X_test)
    for ix,col in enumerate(Y_train.columns):
        y_true = Y_test[col]
        y_pred = Y_pred[:,ix]
        target_names = ['0', '1']
        print(f"====={col}================")
        print(classification_report(y_true, y_pred))

    return model


def export_model(model):
    """
    Export trained model as a pickle file

    INPUTS:
        model:  GridSearchCV - trained grid search object which has "fit" and "predict" methods

    OUTPUTS;
        None

    Results in creation of a pickle file for model
    """
    with open('msg_model_pkl', 'wb') as files:
        pickle.dump(model, files)
    pass



def run_pipeline(messages_file: str, categories_file: str) -> None:
    """
    Takes data related to messages during a crisis and exports a trained model to a pickle file.
    The provided data are joined into a single table and stored in a sqlite database.

    INPUTS:
        messages_file: str - csv file name containing messages
        categories_file: str - csv file containing categories for above messages.

    OUTPUTS:
        None

    -- two files are created as a result of this process:
        messages.sqlite - sqlite database
        message_cat.pkl - pickle file for trained model
    """
    X, y = load_data(messages_file, categories_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    messages_file = sys.argv[1]  # get filename of messages
    categories_file = sys.argv[2]  # get filename of categories
    run_pipeline(messages_file, categories_file)  # run data pipeline