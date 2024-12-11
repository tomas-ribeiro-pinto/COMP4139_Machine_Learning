import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

all_df = pd.read_excel('TrainDataset2024.xls', index_col=False)
all_df.drop('ID', axis=1, inplace=True)
all_df.head()


from sklearn.impute import SimpleImputer

# Replace missing values with median of the column
imputer = SimpleImputer(strategy="median", missing_values=999)
all_df[:] = imputer.fit_transform(all_df)

# classification target
clf_y = all_df['pCR (outcome)']
# regression target
rgr_y = all_df['RelapseFreeSurvival (outcome)']

# Outlier removal approach by:
# Thanaki, Jalaj. Machine Learning Solutions : Expert Techniques to Tackle Complex Machine Learning Problems Using Python, Packt Publishing, Limited, 2018. 
# ProQuest Ebook Central, Available at: http://ebookcentral.proquest.com/lib/nottingham/detail.action?docID=5379696.

# Outlier detection using the following methods:
# 1. Percentile based outlier detection
# 2. MAD (median absolute deviation) based outlier detection
# 3. Standard deviation based outlier detection

import numpy as np

""" 
    Get all the data points that lie under the percentile range from 2.5 to 97.5
"""
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

"""
    Get all the data points that lie under a threshold of 3.5 using modified Z-score (based on the median absolute deviation)
"""
def mad_based_outlier(points, threshold=3.5):
    points = np.array(points)
    if len(points.shape) == 1:
        points = points[:, None]
    median_y = np.median(points)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in points])
    # Small constant added to avoid division by zero
    modified_z_scores = [0.6745 * (y - median_y) / (median_absolute_deviation_y + 1e-6) for y in points]

    return np.abs(modified_z_scores) > threshold

"""
    Get all the data points that lie under a threshold of 3 using standard deviation
"""
def std_div(data, threshold=3):
    std = data.std()
    mean = data.mean()
    isOutlier = []
    for val in data:
        if abs(val - mean)/std > threshold:
            isOutlier.append(True)
        else:
            isOutlier.append(False)
    return isOutlier

"""
    Perform an outlier voting system to determine if a data point is an outlier. 
    If two of the three methods agree that a data point is an outlier, then it is removed.
"""
def outlierVote(data):
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)
    temp = list(zip(x, y, z))
    final = []
    for i in range(len(temp)):
        if temp[i].count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final

def removeOutliers(data):
    # Remove outliers from the dataframe
    for column in data.columns:
        outliers = outlierVote(data[column])
        # Calculate Non-Outlier Maximum and minimum using the outliers list
        non_outlier_max = data.loc[~np.array(outliers), column].max()
        non_outlier_min = data.loc[~np.array(outliers), column].min()

        # Replace outliers with the maximum or minimum non-outlier value
        for i, outlier in enumerate(outliers):
            if outlier:
                data.loc[i, column] = non_outlier_max if data.loc[i, column] > non_outlier_max else non_outlier_min

# See the outlier_removal.py file for the implementation of the function
# Remove outliers, assign modified features to X and drop the outcome columns
removeOutliers(all_df)
X = all_df.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)'], axis=1)

from sklearn.preprocessing import StandardScaler

# Standardise features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# rnd_clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
# rnd_clf.fit(Xs, rgr_y)
rnd_rgr = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
rnd_rgr.fit(Xs, rgr_y)

importances = rnd_rgr.feature_importances_
selected_indices = np.argsort(importances)[::-1]

mandatory_features_indices = [1,3,10]

top_features_indices = [i for i in selected_indices if i not in mandatory_features_indices][:30]

top_features = Xs[:, top_features_indices + mandatory_features_indices]

print(top_features.shape)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Define the model
rnd_forest = RandomForestRegressor(random_state=42)

param_grid = {
  'max_depth': [1,2,3,4,5,6,7,8,9,10],
  'n_estimators': [50, 75, 100],
  'max_features': ['sqrt'],
  'min_samples_split': [2, 5, 7, 10],
  'min_samples_leaf': [1, 2, 4]
}

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(top_features, rgr_y, test_size=0.2, random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=rnd_forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rnd_forest = grid_search.best_estimator_


# Predict the test set
y_pred = best_rnd_forest.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)