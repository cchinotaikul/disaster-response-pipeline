# Disaster Response Message Classifier by Machine Learning

This repository contains the files including raw data of messages sent out in tweets during disaster events which have been cleaned, translated and classified based on the type of messages (e.g., request for help, offer to help, food, and shelter) provided by Figure Eight (https://www.figure-eight.com/), as well as Python scripts used to process and model the classifiers, and
a web app created with Flask. The project is motivated by effort to assist in disaster reliefs by providing a tool to classify related messages.

![Mobile Phone in Disaster Relief](https://github.com/cchinotaikul/disaster-response-pipeline/blob/master/images/disaster_relief_phone.jpg?raw=true)<br>
_[Source](http://www.aidforum.org/topics/mobile-for-development/the-use-of-mobiles-in-disasters/)_

Created as part of my project for the Udacity Data Science Nanodegree course

## Installation

Data analysis and web app created using Python 3.8.5. Libraries used include:
- pandas (1.1.5)
- numpy (1.19.2)
- sklearn (0.23.2)
- sqlalchemy (1.3.20)
- plotly (4.14.3)
- nltk (3.5)
- flask (1.1.2)
- joblib (1.0.0)

## Project Motivation

During modern times of crisis, the widespead use of smartphones allows those involved in the disaster, their friends and relatives, and parties who are looking to assist a large amount of messsages would be sent out. However; this can present a challenge for those involved in the relief efforts to filter for messages that are most relevant to be immediately acted upon. A classifier which allows response teams to quickly categorize messages would be greatly beneficial to this end.

## File Descriptions

Files in this repository include:

<ul>
  <li>\app
    <ul>
      <li>\ templates
          <ul>
            <li>go.html - Webpage for result</li>
            <li>master.html - Main webpage for web app</li>
          </ul>
      </li>
      <li>run.py - script for Flask to generate web app</li>
    </ul>
  </li>
  <li>\data
    <ul>
      <li>disaster_categories.csv - CSV file with categories dataset</li>
      <li>disaster_messages.csv - CSV file with messages dataset</li>
      <li>DisasterResponse.db - Database file created after data is processed</li>
      <li>process_data - Python script for ETL data process</li>
    </ul>
  </li>
  <li>\images
    <ul>
      <li>disaster_relief_phone.jpg - Image for readme header</li>
    </ul>
  </li>
  <li>\models
    <ul>
      <li>train_classifier.py - Machine learning model-generating script</li>
    </ul>
  </li>
</ul>

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` _(This step can be skipped as the repository already includes the db file, unless a new dataset is provided)_
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (Or http://127.0.0.1:3001/)

## Project Summary

The project is composed of the following key steps:

1. An ETL (Extract, Transform and Load) pipeline that reads the provided CSV file and saves as an SQL database
2. A machine learning pipeline that uses count vectorizer, TF-IDF transformer and classify the data using random forest classifier and multi-output classifiers to classify the data based on the given categories. The model is saved as a pickle file
3. A web app created with Flask which uses the model to predict the categories of new messages, as well as display visualisations of the source data file created with Plotly

With the default parameters for the implementations in the sklearn package, the performance of the model is quite satisfactory, with good (over 0.9) precision, accuracy and F1 scores for most of the categories.

Further potential adjustments to the parameters were explored using grid search.

## Acknowledgement and Licensing

- Training/Testing dataset provided by Figure Eight (https://www.figure-eight.com/)
- Libraries' documentations
- Udacity course materials
- All files in this depository are free to use
