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
nltk.download(['punkt', 'wordnet'])

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


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.iloc[:, -36:]

    return X, Y, Y.columns


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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
    # Build pipeline including vectorizer, TF-IDF transformer and
    # random forest classifier with multi-output classifier for
    # predicting multiple target variables
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # Use model to predict Y based on X_test
    predicted = model.predict(X_test)
    predicted_df = pd.DataFrame(predicted, columns=Y_test.columns)

    # Evaluate model on Y_test over each category
    for category in category_names:
        print('Category:', category)
        print(classification_report(Y_test[category], predicted_df[category]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
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
    main()
