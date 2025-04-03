# Disaster-Tweets-Kaggle-Mini-Project

# Disaster Tweets Classification with Word2Vec+BiLSTM

## Project Overview
This project tackles Kaggle's [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) competition, developing a binary classifier to determine whether tweets are about real disasters or not.

## Key Results
| Model Architecture | Public Score | Improvement |
|--------------------|-------------|------------|
| Word2Vec+BiLSTM | 0.78884 | +0.00582% |
| Raw Text Baseline | 0.79344 | +1.33% |
| Advanced Text Cleaning | 0.79190 | +1.13% |
| Basic Text Cleaning | 0.78302 | - |
| CNN+LSTM | 0.78057 | -0.31% |

## Implementation Highlights

### Data Characteristics
- Training set: 7,613 tweets (with target labels)
- Test set: 3,263 tweets
- Class distribution: ~43% disaster, 57% non-disaster

### Model Architecture
```python
Sequential([
    Embedding(10000, 300, weights=[embedding_matrix], trainable=False),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
