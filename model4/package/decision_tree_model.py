from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from .vectorizers import get_count_vectorizer

class DecisionTreeSentiment:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train, params=None, cv=3):
        """Train DecisionTree with optional hyperparameter tuning."""

        pipeline = Pipeline([
            ('vec', get_count_vectorizer()),
            ('clf', DecisionTreeClassifier(random_state=self.random_state))
        ])

        if params is None:
            params = {
                'vec__ngram_range': [(1,1), (1,2)],
                'vec__min_df': [1, 2, 3],
                'vec__max_features': [2000, 5000, None],
                'clf__criterion': ["gini", "entropy"],
                'clf__max_depth': [6, 10, 15, 20],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 3, 5]
            }

        grid = GridSearchCV(
            pipeline,
            params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_cv_score = grid.best_score_

        return self.best_params, self.best_cv_score

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc, 1 - acc
