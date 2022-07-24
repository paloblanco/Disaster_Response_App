# import packages
import sys
from typing import Tuple, List

# data loading and export
import pandas as pd
import sqlite3

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords','omw-1.4'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Reads in raw messages data and joins it, returning a dataframe
    
    INPUTS:
        messages_filepath: str - csv file name containing messages
        categories_filepath: str - csv file containing categories for above messages.

    OUTPUTS:
        df: pd.DataFrame - dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans messages and categories dataframe

    INPUTS:
        df: pd.DataFrame - dataframe containing messages and categories

    OUTPUTS:
        df: pd.DataFrame - dataframe containing messages and categories, cleaned
    """
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
    return df

# this is defined globally so it does not need to be compiled repeatedly in a function
PATTERN_REMOVE_STOPWORDS = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

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

def execute_SVD_get_labels(df):
    """
    This will execute SVD on the comments themselves so we can plot them in two dimensions

    INPUTS:
        df: pd.DataFrame - cleaned dataframe containing messages and categories

    OUTPUTS:
        df_svd: pd.DataFrame - dataframe containing messages, xy coordinates, and a string with all categories
    """
    cv = CountVectorizer(tokenizer=tokenize)
    tf = TfidfTransformer()
    X = tf.fit_transform(cv.fit_transform(df.message))
    svd=TruncatedSVD(n_components=2)
    xplot = svd.fit_transform(X)

    cat_labels = list(df.columns)[4:]
    labels = list(df.apply(lambda x: ', '.join([l for l in cat_labels if x[l]==1]), axis=1))
    
    df_svd = pd.DataFrame()
    df_svd['x'] = xplot[:,0]
    df_svd['y'] = xplot[:,1]
    df_svd['message'] = list(df.message)
    df_svd['labels'] = labels

    return df_svd



def save_data(df: pd.DataFrame, database_filename: str, table_name: str = "messages_classified") -> None:
    """
    Puts messages data into database.

    INPUTS:
        df: pd.DataFrame - dataframe containing messages and categories
        database_filename: str - name of database
        table_name: str - table that will hold data

    OUTPUTS:
        None
    
    -- A file is created as a result of this process:
        database_filename - sqlite database
        This file is overwritten if it exists
    """
    cnxn = sqlite3.connect(database_filename)
    df.to_sql(table_name,cnxn,index=False, if_exists='replace')
    cnxn.close()


def main():
    """
    Loads raw data, cleans it, and saves to a sqlite database.

    INPUTS:
        None

    OUTPUTS: 
        None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Performing SVD Decomp...')
        df_svd = execute_SVD_get_labels(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name="messages_classified")
        save_data(df_svd, database_filepath, table_name="messages_svd")
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    """
    Loads raw data, cleans it, and saves to a sqlite database.

    ARGS:
        messages_filepath - name of csv file containing messages
        categories_filepath - name of csv file containing categories
        database_filepath - name of sqlite database to be (over)written

    OUTPUTS:
        None

    FILES CREATED:
        database_filepath - sqlite database containing joined messages and categories in a single table
    """
    main()