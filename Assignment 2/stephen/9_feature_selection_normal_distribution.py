'''
Results form 2_test_imputed_base_line.py shows that gene predicted dataset performs the best on baseline models
Using this dataset to perform feature selection
'''
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def check_normality(data, alpha=0.001):
    """
    Comprehensive check for normality of features using multiple methods.
    
    Parameters:
    data: pandas DataFrame or numpy array
    alpha: significance level for statistical tests
    
    Returns:
    dict: Results of normality tests for each feature
    """
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    results = {}
    results_short = {}
    
    for column in data.columns:
        feature_data = data[column].dropna()
        
        # 1. Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(feature_data)
        
        # 2. D'Agostino-Pearson test
        agostino_stat, agostino_p = stats.normaltest(feature_data)
        
        # 3. Basic statistics
        skewness = stats.skew(feature_data)
        kurtosis = stats.kurtosis(feature_data)
        
        # Store results
        results[column] = {
            'shapiro_test': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > alpha
            },
            'agostino_test': {
                'statistic': agostino_stat,
                'p_value': agostino_p,
                'is_normal': agostino_p > alpha
            },
            'skewness': skewness,
            'kurtosis': kurtosis
        }

        results_short[column] = bool(shapiro_p > alpha) or bool(agostino_p > alpha)
    
    return results, results_short

def get_normal_cols(normality_results):
  normal_cols = []
  for col, tests in normality_results.items():
    if tests['shapiro_test']['is_normal'] or tests['agostino_test']['is_normal']:
      normal_cols.append(col)
  return normal_cols

if __name__ == '__main__':
    source = 'imputed_csv/gene_predicted_rest_median_imputed.csv'
    df = pd.read_csv(source)

    working_df = df.copy()

    mri_data = working_df.iloc[:, 11:]

    alpha = 0.01 # very loose

    normal_cols = []
    for col in mri_data.columns:
        res = stats.normaltest(mri_data[col])
        if res.pvalue > alpha:
            normal_cols.append(col)
    
    normal_mri_only = working_df.iloc[:, :11]
    for col in normal_cols:
        normal_mri_only[col] = df[col]
    
    normal_mri_only.to_csv('normalized_mri/normal_mri_cols_only.csv', index=False)

    # pca
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    norm_mris = normal_mri_only.iloc[:, 11:]
    scaler = StandardScaler()

    norm_mris = scaler.fit_transform(norm_mris)

    pca = PCA(n_components=2)

    pca_mris = pca.fit_transform(norm_mris)
    norm_mris = normal_mri_only.iloc[:, :11]
    norm_mris['pca_1'] = pca_mris[:, 0]
    norm_mris['pca_2'] = pca_mris[:, 1]

    norm_mris.to_csv('normalized_mri/normal_mri_pca.csv', index=False)




