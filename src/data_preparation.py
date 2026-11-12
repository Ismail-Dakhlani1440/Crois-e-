import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW, PROCESSED, TARGET, RANDOM_STATE, NUMERIC


def prepare_data():
    df = pd.read_csv(RAW).drop(columns='Student_ID')
    df['Social_Media_Hours'] = df['Social_Media_Hours'].clip(upper=df['Screen_Time_Hours'])

    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE)

    q1, q3 = train[NUMERIC].quantile(0.25), train[NUMERIC].quantile(0.75)
    iqr = q3 - q1
    for part in (train, test):
        part[NUMERIC] = part[NUMERIC].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr, axis=1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / 'train.csv', index=False)
    test.to_csv(PROCESSED / 'test.csv', index=False)