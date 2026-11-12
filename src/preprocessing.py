from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from src.config import NUMERIC, ORDINAL, ORDINAL_CATEGORIES, NOMINAL


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES,
                                   handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    onehot_encoder = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)

    return ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC),
        ('ord', ordinal_pipeline, ORDINAL),
        ('nom', onehot_encoder, NOMINAL),
    ])