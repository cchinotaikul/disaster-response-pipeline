import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages data and categories data from CSV files and merge into
    single pandas DataFrame

    Parameters:
        messages_filepath (str): filepath for messages CSV
        categories_filepath (str): filepath for categories CSV

    Output:
        df (pandas DataFrame): merged DataFrame
    '''
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
    Clean the merged DataFrame, including creating columns with Boolean values
    for each message category and removing the original category columns,
    dropping duplicates, and correcting apparently erroneous data.

    Parameter:
        df (pandas DataFrame): DataFrame from load_data() function

    Output:
        df (pandas DataFrame): Cleaned DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # Get column names from the first row (content of string in each column
    # apart from last 2 characters
    row = categories.iloc[0]
    category_colnames = row.apply(lambda col: col[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda row: row[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)

    # drop duplicates
    df = df.drop_duplicates()

    # There are certain messages in related column marked as 2
    # Appears to be error, therefore changed to 0
    df['related'] == df['related'].apply(lambda x: 0 if x == 2 else x)

    return df


def save_data(df, database_filename):
    '''
    Save DataFrame as a SQL database

    Parameters:
        df (pandas DataFrame): DataFrame to be saved
        database_filename: filepath for the saved file (including file
            extension)

    Output:
        None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath,
         database_filepath) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
