Data Projects (TripleTen)
# Project_14_TT
Machine learning for Texts. Natural Language Processing
Project-14: Automated Detection of Negative Movie Reviews Using Machine Learning

## Project Overview

The **Film Junky Union** is developing a sentiment analysis system to classify movie reviews from IMDB as either positive or negative. The goal is to automatically detect negative reviews to help filter and categorize movie feedback. This project applies various machine learning models to classify movie reviews based on polarity labels (positive or negative). The objective is to build a robust classification model that achieves an **F1 score** of at least **0.85** on the test dataset.

## Project Description

This project leverages a dataset of IMDB movie reviews that are labeled as either positive (1) or negative (0). The task is to develop a model capable of predicting the polarity of a given review. 

To achieve this, we followed these main steps:

1. **Data Preprocessing**: The data is cleaned by normalizing the review texts (removing unwanted characters, lowering cases, etc.) and splitting the data into training and testing sets. We used both TF-IDF vectorization and tokenization for feature extraction.

2. **Modeling**: We implemented multiple machine learning models, including:
   - **Logistic Regression**
   - **LightGBM Classifier**
   - **RandomForest Classifier**

   We evaluated each model based on its performance, using the **F1 score** as the main metric.

3. **Evaluation**: The models were tested against a separate test dataset. We also applied the trained models to predict the polarity of 10 new movie reviews, as part of the project instructions.

4. **Results**: We successfully achieved the target F1 score of **0.85** with both the Logistic Regression and LightGBM Classifier models, demonstrating their effectiveness in automatically detecting negative movie reviews.

## Key Features

- **Data Preprocessing**: The dataset was preprocessed using text normalization, tokenization, stopword removal, and lemmatization. TF-IDF vectorization was then applied to convert text into numerical features.
- **Modeling**: Multiple classification models were trained, with the top-performing models being Logistic Regression and LightGBM.
- **Evaluation**: Model performance was assessed using F1 score, with all models achieving satisfactory results above the target of 0.85.
- **Review Prediction**: In addition to model evaluation, the project includes predictions for a set of 10 manually provided movie reviews.

## Data Description

The dataset `imdb_reviews.tsv` contains the following columns:
- **review**: The review text.
- **pos**: The target polarity, where `0` indicates a negative review and `1` indicates a positive review.
- **ds_part**: A field indicating whether the data is part of the `train` or `test` set.

The data was provided by **Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (2011)** and is part of the **Learning Word Vectors for Sentiment Analysis** dataset.

## Requirements

To run this project, the following Python libraries are required:

- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **matplotlib** & **seaborn**: For visualizations.
- **nltk**: For natural language processing (tokenization, stopword removal, and lemmatization).
- **sklearn**: For building and evaluating machine learning models.
- **lightgbm**: For training the LightGBM classifier.
- **spacy**: For advanced tokenization (optional).
- **torch** & **transformers**: Optional, for using BERT embeddings (if you choose to experiment with BERT).

To install the dependencies, run:

pip install pandas numpy matplotlib seaborn nltk scikit-learn lightgbm spacy torch transformers

Conclusion
In this project, we developed a machine learning model to automatically detect negative movie reviews from IMDB. By preprocessing the data, applying text normalization, tokenization, and TF-IDF vectorization, and training multiple classification models, we were able to achieve an F1 score of 0.88 with Logistic Regression and 0.86 with LightGBM. Both models successfully exceeded the target F1 score of 0.85.

Key findings include:

Logistic Regression and LightGBM Classifier were the top-performing models.
Models were evaluated based on F1 score, with all models performing satisfactorily.
We were able to classify new movie reviews with the trained models.
This project demonstrates the effectiveness of machine learning techniques in sentiment analysis and automatic classification of movie reviews.
