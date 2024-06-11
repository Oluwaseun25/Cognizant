# module_train_model.py

# --- 1) IMPORTING PACKAGES ---
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# --- 2) DEFINE GLOBAL CONSTANTS ---
DATA_PATH = "engineered_data.csv"
TARGET = "estimated_stock_pct"
SPLIT = 0.8
SEED = 42

# --- 3) ALGORITHM CODE ---
def load_data(path=DATA_PATH):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param path: str, relative path of the CSV file
    :return: pd.DataFrame
    """
    df = pd.read_csv(path)
    return df

def create_target_and_predictors(df, target=TARGET):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param data: pd.DataFrame, dataframe containing data for the model
    :param target: str, target variable that you want to predict
    :return: pd.DataFrame, pd.Series
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def scale_features(X_train, X_test):
    """Scales the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    This function takes the predictor and target variables and
    trains a specified model using cross-validation, then evaluates it.

    :param X: pd.DataFrame, predictor variables
    :param y: pd.Series, target variable
    :param model: estimator object, the machine learning model to train
    :param model_name: str, name of the model for printing results
    :return: None
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2

def cross_validate_model(model, X, y):
    """Performs cross-validation and returns the average MAE."""
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    avg_mae = -scores.mean()
    return avg_mae

def main():
    """Main function to load data, train and evaluate models."""
    # Load data
    df = load_data()
    
    # Drop non-useful columns
    df.drop(columns=['product_id'], inplace=True)
    
    # Create target and predictors
    X, y = create_target_and_predictors(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=SEED)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Initialize models
    models = {
        "RandomForest": RandomForestRegressor(random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
        "LinearRegression": LinearRegression()
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        mae, r2 = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        avg_mae = cross_validate_model(model, X, y)
        results[name] = {
            "MAE": mae,
            "R2": r2,
            "Avg_CV_MAE": avg_mae
        }
        print(f"{name} - MAE: {mae:.4f}, R2: {r2:.4f}, Avg CV MAE: {avg_mae:.4f}")
    
    # Save the best model (based on lowest MAE)
    best_model_name = min(results, key=lambda k: results[k]['MAE'])
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # Print results
    print("\nFinal Model Performance:")
    for name, metrics in results.items():
        print(f"{name} - MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}, Avg CV MAE: {metrics['Avg_CV_MAE']:.4f}")

if __name__ == "__main__":
    main()
