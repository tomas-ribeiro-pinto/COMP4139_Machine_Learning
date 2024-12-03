import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# regressors 
from sklearn.ensemble import RandomForestRegressor
rnd_forest = RandomForestRegressor(n_estimators=100, random_state=42)

from sklearn.svm import SVR
svr = SVR(C=6, gamma=0.1, kernel='rbf')

from sklearn.neural_network import MLPRegressor
mlp_reg = MLPRegressor(random_state=1, max_iter=300)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=1000)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
#######

dir = 'correlation/'
scaler = StandardScaler()

socres = {}
for file in os.listdir(dir):
    if not file.endswith('.csv'):
        continue
    file_path = os.path.join(dir, file)
    file_name = file.replace('.csv', '')
    df = pd.read_csv(file_path)

    df.drop(columns=['ID'], inplace=True)
    y = df['RelapseFreeSurvival (outcome)']
    X = df.drop(columns=['RelapseFreeSurvival (outcome)'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rnd_forest.fit(X_train_scaled, y_train)
    rnd_pred = rnd_forest.predict(X_test_scaled)
    rnd_mae = mean_absolute_error(y_test, rnd_pred)
    rnd_rmse = root_mean_squared_error(y_test, rnd_pred)
    rnd_r2 = r2_score(y_test, rnd_pred)

    svr.fit(X_train_scaled, y_train)
    svr_pred = svr.predict(X_test_scaled)
    svr_mae = mean_absolute_error(y_test, svr_pred)
    svr_rmse = root_mean_squared_error(y_test, svr_pred)
    svr_r2 = r2_score(y_test, svr_pred)

    mlp_reg.fit(X_train_scaled, y_train)
    mlp_pred = mlp_reg.predict(X_test_scaled)
    mlp_mae = mean_absolute_error(y_test, mlp_pred)
    mlp_rmse = root_mean_squared_error(y_test, mlp_pred)
    mlp_r2 = r2_score(y_test, mlp_pred)

    lin_reg.fit(X_train_scaled, y_train)
    lin_pred = lin_reg.predict(X_test_scaled)
    lin_mae = mean_absolute_error(y_test, lin_pred)
    lin_rmse = root_mean_squared_error(y_test, lin_pred)
    lin_r2 = r2_score(y_test, lin_pred)

    tree_reg.fit(X_train_scaled, y_train)
    tree_pred = tree_reg.predict(X_test_scaled)
    tree_mae = mean_absolute_error(y_test, tree_pred)
    tree_rmse = root_mean_squared_error(y_test, tree_pred)
    tree_r2 = r2_score(y_test, tree_pred)


    local_score = {
        "Random Forest": {
            "MAE": rnd_mae,
            "RMSE": rnd_rmse,
            "R2": rnd_r2
        },
        "SVR": {
            "MAE": svr_mae,
            "RMSE": svr_rmse,
            "R2": svr_r2
        },
        "MLP": {
            "MAE": mlp_mae,
            "RMSE": mlp_rmse,
            "R2": mlp_r2
        },
        "Linear Regression": {
            "MAE": lin_mae,
            "RMSE": lin_rmse,
            "R2": lin_r2
        },
        "Decision Tree": {
            "MAE": tree_mae,
            "RMSE": tree_rmse,
            "R2": tree_r2
        }
    }

    socres[file_name] = local_score


import json
with open('correlation/scores.json', 'w') as f:
    json.dump(socres, f, indent=4)