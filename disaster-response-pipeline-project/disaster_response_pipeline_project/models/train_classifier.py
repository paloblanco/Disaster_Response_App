# import packages
import sys
from tkinter import Grid
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


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    Loads data from specified database and returns inputs and output data for model

    INPUTS:
        database_filepath: str - name of database
    
    OUTPUTS:
        X: pd.Series - series of messages, dtype = text
        Y: pd.DataFrame - dataframe of all category labels corresponding to X
        category_names: List[str] - list of category names
    """
    cnxn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM messages_classified",cnxn)
    cnxn.close()

    X = df.message
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    return X, Y, category_names


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
        'clf__estimator__n_estimators': [5, 10] # 40],
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: list) -> None:
    """
    Print model results by category. Returns Nothing.

    INPUTS:
        model: GridSearchCV - trained model to be used for predict
        X_test: pd.DataFrame - Test input data
        Y_test: pd.DataFrame - Truth output data
        category_names: list - names of cataegories

    OUTPUTS:
        None
    """
    Y_pred = model.predict(X_test)
    for ix,col in enumerate(category_names):
        y_true = Y_test[col]
        y_pred = Y_pred[:,ix]
        target_names = ['0', '1']
        print(f"====={col}================")
        print(classification_report(y_true, y_pred))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Export trained model as a pickle file

    INPUTS:
        model:  GridSearchCV - trained grid search object which has "fit" and "predict" methods
        model_filepath: str - name of pickle file to be (over)written

    OUTPUTS:
        None

    Results in creation of a pickle file for model
    """
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)
    pass


def main():
    """
    Loads cleaned data from database, builds,trains, and saves a model.

    INPUTS:
        None

    OUTPUTS: 
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    """
    Loads cleaned data from database, builds,trains, and saves a model.

    ARGS:
        database_filepath - name of database containing messages
        model_filepath - model to be (over)written

    OUTPUTS: 
        None

    FILES WRITTEN:
        model_filepath - pickle file containing a trained classifier for messages
    """
    main()