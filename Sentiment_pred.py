import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

df_train = pd.read_csv('~/Notebook/Project_14/moved_imdb_reviews_small_lemm_train.tsv', sep='\t')
df_train.info()

df_test = pd.read_csv('~/Notebook/Project_14/moved_imdb_reviews_small_lemm_test.tsv', sep='\t')
df_test.info()

features = df_train['review_lemm']
target = df_train['pos']

# Train the TF-IDF vectorizer and transform the training data
vectorizer = TfidfVectorizer()
features_tfidf = vectorizer.fit_transform(features)

features_tfidf_train, features_tfidf_valid, target_train, target_valid = train_test_split(features_tfidf, target, test_size=0.2, random_state=1234)

model=LogisticRegression(random_state=54321, solver = 'liblinear')
model.fit(features_tfidf_train, target_train)

train_score = model.score(features_tfidf_train, target_train)
valid_score = model.score(features_tfidf_valid, target_valid)
print(f'train_score: {train_score:.4f}')
print(f'valid_score: {valid_score:.4f}')

predict_valid = model.predict(features_tfidf_valid)
accuracy_valid = accuracy_score(target_valid, predict_valid)    # we measured accuracy_score in addition to model.score
print(f'Accuracy on validation set: {accuracy_valid:.4f}')

#Test dataframe
# Transform the test data using the same vectorizer
features_tfidf_test = vectorizer.transform(df_test['review_lemm']) 

# Make predictions on the test set
predict_test = model.predict(features_tfidf_test)

# Adding predictions to the test dataframe
df_test['pos'] = predict_test

df_test.head()

# Save the test_data DataFrame to a CSV file
df_test.to_csv('prediction', index=False)
