from catboost import CatBoostClassifier, Pool


class EnsembleTreeClassifier:
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(**kwargs)

    def fit(self, train_pool: Pool, eval_set: Pool, early_stopping_rounds: int = 50):
        self.model.fit(
            train_pool, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
