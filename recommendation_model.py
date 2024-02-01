# recommendation_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(train_data):
    # Example: Train a simple Random Forest Regressor for rating prediction
    features = ['MovieID', 'WatchedBefore']
    X_train = train_data[features]
    y_train = train_data['Rating']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, test_data):
    # Example: Evaluate the model on the test data
    features = ['MovieID', 'WatchedBefore']
    X_test = test_data[features]
    y_test = test_data['Rating']

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {'Mean Squared Error': mse}
