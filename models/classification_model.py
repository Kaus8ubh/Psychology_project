import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


class RandomForestPipeline:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.label_encoder_X = LabelEncoder()
        self.label_encoder_y = LabelEncoder()
        self.model = None

    def load_and_prepare_data(self, filepath):
        df = pd.read_csv(filepath)

        # Assign labels
        df['parenting_style'] = df[['authoritative_score', 'authoritarian_score', 'permissive_score']].idxmax(axis=1).str.replace('_score', '')
        df['attachment_style'] = df[['secure_score', 'avoidant_score', 'anxious_score']].idxmax(axis=1).str.replace('_score', '')

        X = self.label_encoder_X.fit_transform(df['parenting_style']).reshape(-1, 1)
        y = self.label_encoder_y.fit_transform(df['attachment_style'])

        return X, y

    def train_with_cv(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        all_reports = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Train Random Forest with class weights
            clf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
            clf.fit(X_train_resampled, y_train_resampled)
            self.model = clf  # Save the last trained model

            y_pred = clf.predict(X_test)
            y_test_labels = self.label_encoder_y.inverse_transform(y_test)
            y_pred_labels = self.label_encoder_y.inverse_transform(y_pred)

            print(f"\nFold {fold} Classification Report:")
            print(classification_report(y_test_labels, y_pred_labels))

            cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.label_encoder_y.classes_)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder_y.classes_,
                        yticklabels=self.label_encoder_y.classes_)
            plt.title(f"Confusion Matrix - Fold {fold}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()

    def save_model(self, path="random_forest_model.pkl"):
        if self.model:
            joblib.dump(self.model, path)
            print(f"✅ Model saved to {path}")
        else:
            print("❌ No model found to save.")


