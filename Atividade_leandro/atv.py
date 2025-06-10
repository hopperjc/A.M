import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)

X = df.drop("income", axis=1)
y = LabelEncoder().fit_transform(df["income"])

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB()
}

metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {metric: {} for metric in metrics}

for name, model in models.items():
    for metric_name, metric_func in metrics.items():
        scores = []
        for train_idx, test_idx in kf.split(X, y):
            X_train = preprocessor.fit_transform(X.iloc[train_idx])
            X_test = preprocessor.transform(X.iloc[test_idx])

            if isinstance(model, GaussianNB):
                X_train = X_train.toarray()
                X_test = X_test.toarray()

            model.fit(X_train, y[train_idx])
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            if metric_name == "roc_auc" and y_proba is not None:
                score = metric_func(y[test_idx], y_proba)
            else:
                if metric_name in ["precision", "recall", "f1"]:
                    score = metric_func(y[test_idx], y_pred, zero_division=0)
                else:
                    score = metric_func(y[test_idx], y_pred)

            scores.append(score)
        results[metric_name][name] = scores

for metric_name, model_scores in results.items():
    print(f"\n=== {metric_name.upper()} ===")
    data = list(model_scores.values())
    stat, p = friedmanchisquare(*data)
    print(f"Friedman statistic = {stat:.4f}, p-value = {p:.4f}")
    if p < 0.05:
        print("→ Diferenças significativas detectadas")
    else:
        print("→ Não há evidências de diferença estatística significativa")

# for metric_name, model_scores in results.items():
#     df_plot = pd.DataFrame(model_scores)
#     df_plot.boxplot()
#     plt.title(f"{metric_name.upper()} - Comparação dos Classificadores")
#     plt.ylabel(metric_name)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
