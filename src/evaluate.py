import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay)

from src.config import PROCESSED, MODEL, REPORTS, TARGET


def evaluate_model():
    test = pd.read_csv(PROCESSED / 'test.csv')
    X, y = test.drop(columns=TARGET), test[TARGET]
    model = joblib.load(MODEL)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='macro'),
        'recall': recall_score(y, y_pred, average='macro'),
        'f1': f1_score(y, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y, y_proba, multi_class='ovr'),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=['Low', 'Medium', 'High'])
    plt.savefig(REPORTS / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()