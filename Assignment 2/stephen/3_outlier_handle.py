import numpy as np
import pandas as pd

# outlier detectors
def percentile_based_outlier(data, threshold=95):
  diff = (100 - threshold) / 2.0
  minval, maxval = np.percentile(data, [diff, 100 - diff])
  return (data < minval) | (data > maxval)

# median absolute deviation
def detect_outliers_mad(df, threshold=3):
    medians = df.median()
    mean_abs_dev = (df - medians).abs().median()
    mod_z_score = (0.6745 * (df - medians) / mean_abs_dev)
    return np.abs(mod_z_score) > threshold

# std deviation based
def std_div_outlier(data, threshold=3):
  std = data.std()
  mean = data.mean()
  is_outlier = []
  for v in data:
    is_outlier.append(True if (v > mean + threshold * std) | (v < mean - threshold * std) else False)
  return is_outlier

def vote(data):
  x = percentile_based_outlier(data)
  y = detect_outliers_mad(data)
  z = std_div_outlier(data)
  temp = list(zip(data.index, x, y, z))
  final = []
  for i in range(len(temp)):
    if temp[i].count(False) >= 2:
      final.append(False)
    else:
      final.append(True)
  return final

if __name__ == '__main__':
  df = pd.read_csv('imputed_csv/gene_predicted_rest_median_imputed.csv')

  non_mri = df.iloc[:, 3:14]
  # print(non_mri.columns)
  
  # 1. fill non-mri outlier with median
  for col in non_mri.columns:
    outliers = vote(non_mri[col])
    print(f'{col} has {sum(outliers)} outliers')
  
  
# basically bas no outliers - ignore this step
'''
Age has 0 outliers
ER has 0 outliers
PgR has 0 outliers
HER2 has 0 outliers
TrippleNegative has 0 outliers
ChemoGrade has 2 outliers
Proliferation has 0 outliers
HistologyType has 0 outliers
LNStatus has 0 outliers
TumourStage has 0 outliers
Gene has 0 outliers
'''

