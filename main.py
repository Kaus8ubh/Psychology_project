from models.classification_model import RandomForestPipeline
from models.regression_model import NeuralNetPipeline
from utils.scatter_plot import plot_parenting_attachment_scatter
import pandas as pd

df = pd.read_csv("scores.csv")

if __name__ == "__main__":
    filepath = "scores.csv"
    # plot_parenting_attachment_scatter(df, ['authoritative_score', 'authoritarian_score','permissive_score'], ['secure_score', 'avoidant_score', 'anxious_score'])

    print("=== RANDOM FOREST PIPELINE ===")
    rf = RandomForestPipeline()
    X_rf, y_rf = rf.load_and_prepare_data(filepath)
    rf.train_with_cv(X_rf, y_rf)
    rf.save_model("random_forest_model.pkl")

    print("\n=== NEURAL NETWORK PIPELINE ===")
    nn = NeuralNetPipeline()
    X_nn, y_nn = nn.load_and_prepare_data(filepath)
    nn.train_with_cv(X_nn, y_nn)
