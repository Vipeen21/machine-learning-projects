
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(input_csv_path, output_dir):
    df = pd.read_csv(input_csv_path)

    # Drop CUST_ID column as it\"s an identifier and not a feature
    df = df.drop("CUST_ID", axis=1)

    # One-hot encode the categorical column
    df = pd.get_dummies(df, columns=["CAT_GAMBLING"], drop_first=True)

    # Define features (X) and target (y)
    X = df.drop("CREDIT_SCORE", axis=1)
    y = df["CREDIT_SCORE"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print("Data preprocessing complete.")
    return X_train.columns

def train_model(data_dir, model_output_path):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    model.save_model(model_output_path)
    print("XGBoost model trained and saved successfully.")

def evaluate_model(data_dir, model_path, plot_output_path):
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Credit Score")
    plt.ylabel("Predicted Credit Score")
    plt.title("Actual vs. Predicted Credit Scores")
    plt.grid(True)
    plt.savefig(plot_output_path)
    plt.close()
    print("Model evaluation complete and visualization saved.")

if __name__ == "__main__":
    input_csv = "/home/ubuntu/upload/credit_score.csv"
    processed_data_dir = "/home/ubuntu/processed_data"
    model_file = "/home/ubuntu/xgboost_credit_score_model.json"
    plot_file = "/home/ubuntu/actual_vs_predicted.png"

    # Preprocess data
    preprocess_data(input_csv, processed_data_dir)

    # Train model
    train_model(processed_data_dir, model_file)

    # Evaluate model
    evaluate_model(processed_data_dir, model_file, plot_file)

    print("Credit score modeling program finished.")


