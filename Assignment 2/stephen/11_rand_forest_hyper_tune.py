import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def hyper_tune_random_forest(src):
    df = pd.read_csv(src)
    df.drop(columns=['ID'], inplace=True)
    y = df['RelapseFreeSurvival (outcome)']
    X = df.drop(columns=['RelapseFreeSurvival (outcome)'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    rnd_forest = RandomForestRegressor(random_state=42)

    # Define the parameter grid
    param_grid = {
        'max_depth': [1,2,3,4,5,6,7,8,9,10],
        'n_estimators': [50, 75, 100, 150, 200],
        'max_features': ['sqrt'],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=rnd_forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_rnd_forest = grid_search.best_estimator_

    # Make predictions
    rnd_pred = best_rnd_forest.predict(X_test_scaled)

    # Evaluate the model
    rnd_mae = mean_absolute_error(y_test, rnd_pred)
    rnd_rmse = root_mean_squared_error(y_test, rnd_pred)
    rnd_r2 = r2_score(y_test, rnd_pred)

    res = {
        'best_params': grid_search.best_params_,
        'mean_absolute_error': rnd_mae,
        'root_mean_squared_error': rnd_rmse,
        'r2_score': rnd_r2
    }

    return res

import json
if __name__ == '__main__':
    files = [
        # 'altered_csv/original.csv',

        'pca_csv/all_pca_1.csv', 
        'pca_csv/all_pca_2.csv', 
        'pca_csv/mri_pca_1.csv', 
        'pca_csv/mri_pca_2.csv',

        'normalized_mri/normal_mri_cols_only.csv',
        'normalized_mri/normal_mri_pca_1.csv',
        'normalized_mri/normal_mri_pca_2.csv',
        'normalized_mri/mandatory_normal_mri_pca_1.csv',
        'normalized_mri/mandatory_normal_mri_pca_2.csv',
        'normalized_mri/norm_pca2_only.csv',


        # 'correlation/correlation_filtered.csv',
        # 'correlation/correlation_filtered.csv',
        # 'correlation/filterd_by_coorelation_pca_1.csv',
        # 'correlation/filterd_by_coorelation_pca_2.csv',

        # 'imputed_csv/gene_predicted_rest_median_imputed.csv',
        # 'imputed_csv/na_pcr_drapped.csv',
        # 'imputed_csv/na_pcr_missing_median_imputed.csv',
        # 'imputed_csv/gene_predicted_rest_median_imputed_no_outliers.csv',

        # 'k_best_csv/k_best_5.csv',
        # 'lasso_csv/lasso_important_features.csv',
    ]
    result = {}
    for file in files:
        best = hyper_tune_random_forest(file)
        result[file] = best
    
    with open('hyper_tune_results/hyper_tune_random_forest_result.json', 'w') as f:
        json.dump(result, f, indent=4)
        