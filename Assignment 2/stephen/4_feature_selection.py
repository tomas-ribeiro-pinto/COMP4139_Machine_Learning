'''
Results form 2_test_imputed_base_line.py shows that gene predicted dataset performs the best on baseline models
Using this dataset to perform feature selection
'''
import pandas as pd
source = 'imputed_csv/gene_predicted_rest_median_imputed.csv'
# source = 'imputed_csv/gner.csv'


df = pd.read_csv(source)
y = df['RelapseFreeSurvival (outcome)']
ID = df['ID']
X = df.drop(columns=['RelapseFreeSurvival (outcome)', 'ID'])

# 1. PCA
# 1.1 PCA on all mri features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

mri_cols = X.columns[12:]

X_mri = X[mri_cols]

scaler = StandardScaler()
X_mri = scaler.fit_transform(X_mri)

pca = PCA(n_components=2)
X_mri_pca = pca.fit_transform(X_mri)

rest_cols = X.columns[:12]
X_rest = X[rest_cols].copy()
X_rest.loc[:, 'mri_pca_1'] = X_mri_pca[:, 0]

X_rest.insert(0, 'ID', ID)
X_rest.insert(2, 'RelapseFreeSurvival (outcome)', y)

X_rest.to_csv(
  'pca_csv/mri_pca_1.csv',
  index=False
)

X_rest.loc[:, 'mri_pca_2'] = X_mri_pca[:, 1]
X_rest.to_csv(
  'pca_csv/mri_pca_2.csv',
  index=False
)

# 1.2 PCA on all features besides from ['ER', 'HER2', 'Gene']
pCR, ER, HER2, Gene = X['ER'], X['ER'], X['HER2'], X['Gene']

all_pca_df = X.drop(columns=['ER', 'HER2', 'Gene'])

scaler = StandardScaler()
all_pca_df = scaler.fit_transform(all_pca_df)

pca = PCA(n_components=2)

all_pca = pca.fit_transform(all_pca_df)

df_fin = pd.DataFrame()
df_fin.insert(0, 'ID', ID)
df_fin.insert(1, 'pCR (outcome)', pCR)
df_fin.insert(2, 'RelapseFreeSurvival (outcome)', y)
df_fin.insert(3, 'ER', ER)
df_fin.insert(4, 'HER2', HER2)
df_fin.insert(5, 'Gene', Gene)
df_fin.insert(6, 'pca_1', all_pca[:, 0])

df_fin.to_csv(
  'pca_csv/all_pca_1.csv',
  index=False
)

df_fin.insert(7, 'pca_2', all_pca[:, 1])

df_fin.to_csv(
  'pca_csv/all_pca_2.csv',
  index=False
)

### K Best Selection ###
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif

from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest

# selector = SelectKBest(k=5)
selector = SelectKBest(score_func=f_regression, k='all')
X_new = selector.fit_transform(X, y)

df_fin = pd.DataFrame()
df_fin.insert(0, 'ID', ID)
df_fin.insert(1, 'pCR (outcome)', pCR)
df_fin.insert(2, 'RelapseFreeSurvival (outcome)', y)
df_fin.insert(3, 'ER', ER)
df_fin.insert(4, 'HER2', HER2)
df_fin.insert(5, 'Gene', Gene)

df_fin.insert(6, 'k_best_1', X_new[:, 0])
df_fin.insert(7, 'k_best_2', X_new[:, 1])
df_fin.insert(8, 'k_best_3', X_new[:, 2])
df_fin.insert(9, 'k_best_4', X_new[:, 3])
df_fin.insert(10, 'k_best_5', X_new[:, 4])

df_fin.to_csv(
  'k_best_csv/k_best_5.csv',
  index=False
)

## remove features with low variance
from sklearn.feature_selection import VarianceThreshold
vt_working = X.copy()
vt = VarianceThreshold(threshold=0.1)  # Minimum variance threshold
# X_reduced = vt.fit_transform(vt_working)
reduction = vt.fit(vt_working)
support_v = vt.get_support()

# compute columns to keep
columns_to_keep = []
for i in range(len(support_v)):
  if support_v[i]:
    columns_to_keep.append(vt_working.columns[i])

df_fin = pd.DataFrame()
df_fin.insert(0, 'ID', ID)
df_fin.insert(1, 'RelapseFreeSurvival (outcome)', y)

for col in columns_to_keep:
  df_fin.insert(len(df_fin.columns), col, vt_working[col])

df_fin.to_csv(
  'k_best_csv/variance_threshold_0.1.csv',
  index=False
)


### Recursive Feature Elimination ###
# wrapper method
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)

df_fin = pd.DataFrame()
df_fin.insert(0, 'ID', ID)
df_fin.insert(1, 'RelapseFreeSurvival (outcome)', y)




df_fin.insert(3, 'ER', ER)
df_fin.insert(4, 'HER2', HER2)
df_fin.insert(5, 'Gene', Gene)




