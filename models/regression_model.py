import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralNetPipeline:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def load_and_prepare_data(self, filepath):
        df = pd.read_csv(filepath)
        X = df[['authoritative_score', 'authoritarian_score', 'permissive_score']].values
        y = df[['secure_score', 'avoidant_score', 'anxious_score']].values
        y_class = np.argmax(y, axis=1)
        return X, y_class

    def build_model(self, input_dim):
        model = models.Sequential([ 
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')  # Softmax output for multi-class classification
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_with_cv(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scaler = StandardScaler()

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            y_train_cat = to_categorical(y_train, 3)
            y_test_cat = to_categorical(y_test, 3)

            # Class weights
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))

            model = self.build_model(input_dim=X.shape[1])
            history = model.fit(X_train_scaled, y_train_cat, validation_split=0.2,
                                epochs=100, verbose=0, class_weight=class_weight_dict)

            # Predict probabilities
            y_pred_probs = model.predict(X_test_scaled)

            # Print probabilities as percentage for each class (attachment style)
            for i, probs in enumerate(y_pred_probs):
                print(f"\nSample {i+1} Prediction Probabilities (Percentage):")
                attachment_styles = ['Secure', 'Avoidant', 'Anxious']
                for j, prob in enumerate(probs):
                    print(f"{attachment_styles[j]}: {prob*100:.2f}%")
            
            # Predict class (max probability) for evaluation
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            print(f"\nFold {fold} Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred_classes, target_names=['Secure', 'Avoidant', 'Anxious']))

            cm = confusion_matrix(y_test, y_pred_classes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Secure', 'Avoidant', 'Anxious'],
                        yticklabels=['Secure', 'Avoidant', 'Anxious'])
            plt.title(f"Confusion Matrix - Fold {fold}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
