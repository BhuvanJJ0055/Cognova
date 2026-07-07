from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VaderFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that extracts VADER sentiment scores
    (compound, pos, neu, neg) from raw text as dense numerical features.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for text in X:
            scores = self.analyzer.polarity_scores(text)
            features.append([scores['compound'], scores['pos'], scores['neu'], scores['neg']])
        return np.array(features)
