import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import PROCESSED, MODEL, TARGET
from src.preprocessing import build_preprocessor


def train_model():
    train = pd.read_csv(PROCESSED / 'train.csv')
    X, y = train.drop(columns=TARGET), train[TARGET]

    model = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', LogisticRegression(C=1, max_iter=1000)),
    ])
    model.fit(X, y)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)