import sys

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sqlalchemy import create_engine
import nltk

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''
    Load data from SQL database, then separate to dependent (Y) and
    independent (X) variable sets. Outputs X and Y datasets and list of
    variables of Y

    Parameter:
        database_filepath (str): filepath of db file

    Returns:
        X (pandas dataframe): the messages to be processed into dependant
            variable for machine learning purpose
        Y (pandas dataframe): flags for categories of the messages
        Y.columns (list): list of Y variables
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.iloc[:, -36:]

    return X, Y, Y.columns


def tokenize(text):
    '''
    Text tokenizing function for messages. Replaces URLs with placeholder,
    remove punctuations, then tokenize, lemmatize, lowercase and strip
    whitespaces from the words. Output cleaned tokens for machine learning

    Parameter:
        text (str): string of single text message in English

    Output:
        clean_tokens (list): list of tokenized words
    '''
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 '[!*[(][)],]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Find URLs in text
    detected_urls = re.findall(url_regex, text)

    # Replace URLs with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove punctuations
    text = re.sub(r"""[,.;@#?!&$]+\ *""", " ", text, flags=re.VERBOSE)

    # Tokenise words
    tokens = word_tokenize(text)
    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    # Lemmatize words, turn to lowercase and strip whitespaces
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # Append to list
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build and output pipeline including vectorizer, TF-IDF transformer and
    random forest classifier with multi-output classifier

    Parameters:
        None

    Output:
        pipeline (sklearn Pipeline): Model pipeline for multi-output
            classification
    '''
    # Build pipeline including vectorizer, TF-IDF transformer and
    # random forest classifier with multi-output classifier for
    # predicting multiple target variables
    pipeline = Pipeline([
        # ngram_range of (1,2), max_features of 10000 produces better model
        # according to grid search
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2),
                                 max_features=10000)),
        # use_idf set to false produces better model according to grid search
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate multi-output model and print classification report for each Y
    output including precision, recall, and F1 scores

    Parameters:
        model (sklearn Pipeline): pipeline including vectorizer, transformer,
            and multi-output classifier
        X_test (pandas DataFrame): input variables for model
        Y_test (pandas DataFrame): actual output variables for model evaluation
        category_names (list): list of output categories

    Output:
        None
    '''
    # Use model to predict Y based on X_test
    predicted = model.predict(X_test)
    predicted_df = pd.DataFrame(predicted, columns=Y_test.columns)

    # Evaluate model on Y_test over each category
    for category in category_names:
        print('Category:', category)
        print(classification_report(Y_test[category], predicted_df[category]))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file

    Parameters:
        model (sklearn Pipeline): model to be saved
        model_filepath (str): filepath for saving (including file extension)

    Output:
        None
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
