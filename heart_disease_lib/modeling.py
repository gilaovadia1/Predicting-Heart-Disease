import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals.array_api_extra import nunique
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import KFold

class DataManager:
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.scaler = None

    def _print_split(self, X_train, y_train, X_test, y_test):
        print('Train/Test Split:\n')
        print(f'X_train: {X_train.shape}, y_train: {y_train.shape}, y_freq: {round(y_train.mean(), 2)}')
        print(f'X_test: {X_test.shape}, y_test: {y_test.shape}, y_freq: {round(y_test.mean(), 2)}')
        print('\n')

    def split (self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state)

        self._print_split(X_train, y_train, X_test, y_test)

        return X_train, X_test, y_train, y_test

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, ohe_columns=None, target_encode_columns=None, interaction_pairs=None):
        self.ohe_columns = ohe_columns or []
        self.target_encode_columns = target_encode_columns or []
        self.interaction_pairs = interaction_pairs or []  # List of tuples: [('col1', 'col2'), ...]

        self.ohe_categories_ = {}
        self.target_encoder_ = None
    
    def fit(self, X, y=None):
        self._fit_ohe(X)
        self._fit_target_encode(X, y)
        return self
    
    def transform(self, X):
        X = X.copy()
        X = self._ohe(X)
        X = self._target_encode(X)
        X = self._interactions(X)
        return X
    
    # --- OHE methods ---
    
    def _fit_ohe(self, X):
        for col in self.ohe_columns:
            if col in X.columns:
                self.ohe_categories_[col] = X[col].unique()
    
    def _ohe(self, X):
        for col in self.ohe_columns:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col, dtype=int)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        return X
    
    # --- Target encoding methods ---
    
    def _fit_target_encode(self, X, y):
        if y is not None and self.target_encode_columns:
            self.target_encoder_ = TargetEncoder()
            self.target_encoder_.fit(X[self.target_encode_columns], y)
    
    def _target_encode(self, X):
        if self.target_encoder_ and self.target_encode_columns:
            X[self.target_encode_columns] = self.target_encoder_.transform(X[self.target_encode_columns])
        return X
    
    # --- Interaction methods ---
    
    def _interactions(self, X):
        """Create interaction features from pairs of columns.
        
        interaction_pairs: List of tuples, e.g. [('col1', 'col2'), ('col3', 'col4')]
        """
        for col1, col2 in self.interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                # Create interaction column with descriptive name
                interaction_name = f"{col1}_x_{col2}"
                X[interaction_name] = X[col1] * X[col2]
        return X

    

class Model:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        #print('Model fitted')

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def fit_and_predict(self, X_train, y_train, X_test, y_test, plot=False, print=False):
        self.fit(X_train, y_train)
        pred = self.predict_proba(X_test)
        ra = ResultAnalyzer(y_test, pred)

        return(ra.roc_curve(plot=plot, print=print))

    def _log_submission(self, submission_path, submission_id=None, model=None, Preprocessing=None, notes=None):
        log_path = rf"{submission_path}/submission_log.txt"

        with open(log_path, "a") as f:
            f.write("\n" + "=" * 40 + "\n")
            f.write(f"Submission: {submission_id}\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Preprocessing: {Preprocessing}\n")
            f.write(f"Notes: {notes}\n")

    def submit(self, X_Valid, submission_num=None, model=None, Preprocessing=None, notes=None):
        submission_id = f"{submission_num}_{str(datetime.today()).split()[0]}"
        submission_path = rf"C:\Users\Admin\Predicting-Heart-Disease\submissions"

        # Predict
        pred_proba_validation = self.predict_proba(X_Valid)

        # Save in desired format
        submission = pd.DataFrame({
            "id": X_Valid['id'],
            "Heart Disease": pred_proba_validation
        })

        submission.to_csv(f'{submission_path}/submission_{submission_id}.csv', index=False)
        self._log_submission(submission_path, submission_id, model, Preprocessing, notes)

        print("Submission saved & logged")

class CrossValidator:
    def __init__(self, model_class, data_manager_class, k_folds=5,
                  scoring='auc'):
        self.model_class = model_class
        self.data_manager_class = data_manager_class
        self.k_folds = k_folds
        self.scoring = scoring
        self.scores = []

    def cross_validate(self, X, y, model, pipeline=None, feature_engineer=None, n_folds=5):
        """Cross-validate a model with optional feature engineering and pipeline preprocessing.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        y : array-like
            Target vector
        model : Model
            Model instance with fit_and_predict method
        pipeline : Pipeline, optional
            Scikit-learn Pipeline for preprocessing (e.g., scaling, encoding)
        feature_engineer : FeatureEngineer, optional
            FeatureEngineer instance to apply before pipeline. 
            Fit separately on each fold's training set.
        n_folds : int
            Number of cross-validation folds (default: 5)
        
        Returns
        -------
        list
            List of scores (AUC) from each fold
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
            X_fold_train, X_fold_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
            y_fold_train, y_fold_test = y[train_index], y[test_index]

            # Step 1: Apply feature engineering (fit on train fold only, then transform both)
            if feature_engineer is not None:
                X_fold_train = feature_engineer.fit(X_fold_train, y_fold_train).transform(X_fold_train)
                X_fold_test = feature_engineer.transform(X_fold_test)
                print(f"[CV Fold {fold_num + 1}] FeatureEngineer applied. Train shape: {X_fold_train.shape}")

            # Step 2: Apply pipeline preprocessing if provided
            if pipeline is None:
                X_train_scaled = X_fold_train
                X_test_scaled = X_fold_test
            else:
                X_train_scaled = pipeline.fit_transform(X_fold_train, y_fold_train)
                X_test_scaled = pipeline.transform(X_fold_test)

            # Step 3: Fit and predict
            score = model.fit_and_predict(X_train_scaled, 
                                         y_fold_train, 
                                         X_test_scaled, 
                                         y_fold_test)
            scores.append(score)
            print(f"[CV Fold {fold_num + 1}] Score: {score:.4f}")

        return scores


class ResultAnalyzer:
    def __init__(self, y_true, y_prob=None):
        """
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities for the positive class (for ROC)
        """
        self.y_true = y_true
        #self.y_pred = y_pred
        self.y_prob = y_prob

    def roc_curve(self, plot=False, print=False):
        """
        Plot ROC curve. Requires predicted probabilities.
        """
        if self.y_prob is None:
            raise ValueError("y_prob (predicted probabilities) required for ROC curve")

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)

        roc_auc = auc(fpr, tpr)

        if plot:
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        # return fpr, tpr, roc_auc

        if print:
            print(f"AUC: {roc_auc}")
        return roc_auc
        