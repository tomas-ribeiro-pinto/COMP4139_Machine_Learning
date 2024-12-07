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

def removeOutliers(df):
    # Remove outliers from the dataframe
    for column in df.columns:
        outliers = outlierVote(df[column])
        # Calculate Non-Outlier Maximum using the outliers list
        non_outlier_max = df.loc[~np.array(outliers), column].max()
        # Replace outliers with the maximum non-outlier value
        df.loc[outliers, column] = non_outlier_max