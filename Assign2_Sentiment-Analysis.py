"""
Sentiment Classification.

This module implements a sentiment classifier to categorize restaurant reviews
into three sentiment classes: negative, neutral, and positive.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample

# Defining File Path
reviews_file_path = "reviews.csv"
train_file_path = "train.csv"
valid_file_path = "valid.csv"

def load_and_preprocess_data(input_file):
    """
    Load & preprocess a tab-separated dataset of reviews.

    This function reads a dataset from a specified input file, processes the
    'RatingValue' column to assign sentiment categories (negative, neutral,
    or positive), and separates the data into three subsets based on sentiment.
    """
    data = pd.read_csv(input_file, delimiter='\t')
    data['Sentiment'] = data['RatingValue'].apply(
        lambda x: 0 if x in [1, 2] else (1 if x == 3 else 2)
    )

    negative_reviews = data[data['Sentiment'] == 0]
    neutral_reviews = data[data['Sentiment'] == 1]
    positive_reviews = data[data['Sentiment'] == 2]
    print("Postive Reviews before sampling", len(positive_reviews))
    print("Negative Reviews before sampling", len(negative_reviews))
    print("Neutral Reviews before sampling", len(neutral_reviews))


    positive_downsampled = resample(positive_reviews,
                                    replace=False,
                                    n_samples=220,
                                    random_state=42)


    neutral_downsampled = resample(neutral_reviews,
                                   replace=False,
                                   n_samples=len(negative_reviews),
                                   random_state=42)


    print("Positive Reviews after sampling", len(positive_downsampled))
    print("Neutral Reviews after sampling", len(neutral_downsampled))


    balanced_data = pd.concat(
        [negative_reviews, neutral_downsampled, positive_downsampled]
    )
    balanced_data = balanced_data.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    balanced_data = balanced_data[['Sentiment', 'Review']]


    return balanced_data


def split_and_save_data(data):
    """
    Split the data into training and validation and save them as CSV.

    This function splits the input dataset into two parts: training set (80%)
    and a validation set (20%), ensuring that the split preserves the class
    distribution
    """
    train_data, valid_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['Sentiment']
    )
    train_data.to_csv(train_file_path, index=False)
    valid_data.to_csv(valid_file_path, index=False)


def train_model(train_file):
    """
    Train a model by transforming text reviews into features using n-grams.

    This function reads the training data from a CSV file, applies an n-gram
    vectorization to the 'Review' column, and prepares the data for model
    training
    """
    train_data = pd.read_csv(train_file)
    vectorizer = CountVectorizer(ngram_range=(2,3))
    x_train = vectorizer.fit_transform(train_data['Review'])
    y_train = train_data['Sentiment']


    model = MultinomialNB(alpha=0.5)
    model.fit(x_train, y_train)


    return model, vectorizer


def evaluate(filename, model, vectorizer):
    """
    Evaluate the performance of a trained model on a validation dataset.

    This function reads the validation data from a CSV file, transforms the
    text reviews into feature vectors using the provided vectorizer, and
    uses the trained model to make predictions.
    """
    data = pd.read_csv(filename)


    x_valid = vectorizer.transform(data['Review'])
    y_valid = data['Sentiment']
    y_pred = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    avg_f1_score = f1_score(y_valid, y_pred, average='macro')
    class_f1_scores = f1_score(y_valid, y_pred, average=None)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average F1 score: {avg_f1_score:.2f}")
    print(f"Class-wise F1 scores:")
    print(f"  Negative: {class_f1_scores[0]:.2f}")
    print(f"  Neutral: {class_f1_scores[1]:.2f}")
    print(f"  Positive: {class_f1_scores[2]:.2f}")
    print(f"Confusion matrix:")
    print(f"           Negative  Neutral  Positive")
    print(f"Negative   {conf_matrix[0][0]}         {conf_matrix[0][1]}        {conf_matrix[0][2]}")
    print(f"Neutral    {conf_matrix[1][0]}         {conf_matrix[1][1]}        {conf_matrix[1][2]}")
    print(f"Positive   {conf_matrix[2][0]}         {conf_matrix[2][1]}        {conf_matrix[2][2]}")


if __name__ == "__main__":
    data = load_and_preprocess_data(reviews_file_path)


    split_and_save_data(data)

    model, vectorizer = train_model(train_file_path)

    evaluate(valid_file_path, model, vectorizer)  # Use this function for evaluation on Unseen Data
