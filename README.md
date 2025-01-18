# Stock Sentiment Analysis

This project analyzes stock market sentiment based on historical data using Natural Language Processing (NLP) techniques and machine learning algorithms. A Random Forest Classifier is used to predict stock movements based on news headlines.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Implementation](#model-implementation)
- [Evaluation](#evaluation)
- [Usage](#usage)

## Dataset
The dataset contains stock market news headlines and corresponding labels:
- **Date**: Date of the news headlines.
- **Headlines**: 25 news headlines for a given date.
- **Label**: Binary labels indicating stock movement (1: Positive, 0: Negative).

Ensure the dataset file (`Data3.csv`) is encoded in `ISO-8859-1` and located in the specified path.

## Preprocessing
1. Split the dataset into training and testing sets:
   - Training set: News before `20150101`.
   - Testing set: News after `20141231`.

2. Clean the text data:
   - Remove punctuations and non-alphabetical characters.
   - Convert all text to lowercase for consistency.

3. Combine all 25 headlines for each date into a single string.

## Feature Extraction
- **Bag of Words (BOW)**:
  - Use `CountVectorizer` with bigrams (`ngram_range=(2,2)`) to transform text data into feature vectors.

## Model Implementation
- **Random Forest Classifier**:
  - Number of estimators: 200.
  - Criterion: Entropy.
  - Train the model on the transformed training dataset.

## Evaluation
- Evaluate the model on the test dataset using:
  - **Confusion Matrix**: Understand the performance of the classifier.
  - **Accuracy Score**: Overall accuracy of the model.
  - **Classification Report**: Precision, recall, and F1-score.

## Usage
### Dependencies
Install the required Python libraries:
```bash
pip install pandas scikit-learn
```

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
   ```
2. Place the dataset file (`Data3.csv`) in the repository folder.

3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

### Example Output
Confusion Matrix:
```
[[25 10]
 [ 5 30]]
```
Accuracy Score:
```
0.85
```
Classification Report:
```
              precision    recall  f1-score   support

           0       0.83      0.89      0.86        35
           1       0.86      0.79      0.82        38

    accuracy                           0.85        73
   macro avg       0.85      0.84      0.84        73
weighted avg       0.85      0.85      0.85        73
```

