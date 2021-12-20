#!/usr/bin/env python3
import time
from loguru import logger
import pandas as pd
import sklearn.feature_selection as fs
import sklearn as sk
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


DATASET = "saved.csv"
LABEL_NAME = "label"


class KerasNNClassifier:
    def __init__(self, dense, dropout, epochs):
        self.epochs = epochs
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(dense, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model.compile(loss="mse", optimizer="adam")

    def fit(self, x, y):
        return self.model.fit(x, y, epochs=self.epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == "__main__":
    # Loading the dataset
    df = pd.read_csv(DATASET, sep=",")
    logger.info("Dataset loaded: {} rows", len(df.index))

    # Binary label column
    y = df[LABEL_NAME]
    y_bin = np.where(y == "normal", 1, 0)  # 1 is normal 0 is attack
    y_bin = LabelEncoder().fit_transform(y_bin)
    # Preprocessing and cleanup
    normal_frame = df.loc[df[LABEL_NAME] == 1]
    logger.info("Normal data points: {} rows", len(normal_frame.index))

    # Drop label column
    X = df.drop(columns=[LABEL_NAME])
    # Drop non-number columns
    X_numbers = X.select_dtypes(exclude=["object"])
    # Impute missing values with 0
    imp = SimpleImputer(missing_values=np.nan, strategy="constant")
    features = X_numbers.columns
    X_numbers = pd.DataFrame(imp.fit_transform(X_numbers), columns=features)

    # Normalize with min-max
    scaler = MinMaxScaler()
    X_number = pd.DataFrame(scaler.fit_transform(X_numbers), columns=features)

    # Feature ranking
    chi_rank = fs.chi2(X_numbers, y)
    rank_df = pd.DataFrame({"label": features, "rank": chi_rank[0]})
    rank_df = rank_df.sort_values(by=["rank"], ascending=False)

    logger.info(
        "Feature ranking top 30:\n{}",
        rank_df[:30].to_csv(index=False, sep=";", float_format="%.2f"),
    )
    # Feature selection
    selector = fs.SelectKBest(fs.chi2, k=30)
    selector.fit_transform(X_numbers, y)
    X_reduced = X_numbers.iloc[:, selector.get_support(indices=True)]

    logger.info("Selected features:\n{}", "\n".join(X_reduced.columns))

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.cluster import KMeans

    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K Nearest Neighbours (k=5)": KNeighborsClassifier(n_neighbors=5),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "KMeans": KMeans(n_clusters=2),
        "Keras Neural Net": KerasNNClassifier(len(features), 0.5, 100),
    }

    results = {}
    for name, model in models.items():
        results[name] = {
            "accuracy": [],
            "recall": [],
            "mcc": [],
            "time": [],
        }
        for i in range(5):
            X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
                X_numbers, y_bin, test_size=0.33, random_state=i
            )

            start = time.perf_counter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end = time.perf_counter()
            accuracy = sk.metrics.accuracy_score(y_test, y_pred) * 100
            recall = sk.metrics.recall_score(y_test, y_pred, pos_label=0) * 100
            mcc = sk.metrics.matthews_corrcoef(y_test, y_pred)
            elapsed = end - start
            logger.info("{} took {:.4f}s", name, elapsed)
            logger.info("{} Accuracy: {:.4f}", name, accuracy)
            logger.info("{} Matthews: {:.4f}", name, mcc)
            logger.info("{} Recall: {:.4f}", name, recall)
            results[name]["accuracy"].append(accuracy)
            results[name]["recall"].append(recall)
            results[name]["mcc"].append(mcc)
            results[name]["time"].append(elapsed)

    res = []
    for name, values in results.items():
        res.append(
            [
                name,
                np.mean(values["accuracy"]),
                np.mean(values["recall"]),
                np.mean(values["mcc"]),
                np.mean(values["time"]),
            ]
        )
    results_df = pd.DataFrame(
        res, columns=["name", "accuracy", "recall", "mcc", "time"]
    )
    logger.info(
        "Final results:\n{}",
        results_df.to_csv(index=False, sep=";", float_format="%.4f"),
    )
