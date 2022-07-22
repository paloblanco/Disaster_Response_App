# import packages
import sys
from typing import Tuple, List

# data loading and export
import pandas as pd
import sqlite3


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


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Puts messages data into database.

    INPUTS:
        df: pd.DataFrame - dataframe containing messages and categories
        database_filename: str - name of database

    OUTPUTS:
        None
    
    -- A file is created as a result of this process:
        database_filename - sqlite database
        This file is overwritten if it exists
    """
    cnxn = sqlite3.connect(database_filename)
    df.to_sql("messages_classified",cnxn,index=False, if_exists='replace')
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
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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