
import pandas as pd

source = 'imputed_csv/gene_predicted_rest_median_imputed.csv'

df = pd.read_csv(source)

min_df = df.copy()
min_df.drop('ID', axis=1, inplace=True)
min_df = min_df.iloc[:, :]
all_coorelation_matrix = min_df.corrwith(min_df['RelapseFreeSurvival (outcome)'])
from pprint import pprint

# save with label
val_matrix = all_coorelation_matrix.sort_values(ascending=False)

# keep columns with abs(correlation) > 0.1
val_matrix = val_matrix[abs(val_matrix) > 0.1]

val_matrix.to_csv('correlation.csv')

matrix = val_matrix.to_dict()

cols = list(matrix.keys())
good_cols = cols[1:]

y = df['RelapseFreeSurvival (outcome)']
fixed_cols = ['ID','pCR (outcome)','RelapseFreeSurvival (outcome)','Gene', 'ER', 'HER2']

# build new dataframe with only good columns and mandatory columns
raw_df = df[fixed_cols + good_cols]
raw_df.to_csv('correlation/correlation_filtered.csv', index=False)

### pca
non_mandatory = [col for col in raw_df.columns if col not in fixed_cols]
pca_raw = raw_df[non_mandatory]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pca_raw_scaled = scaler.fit_transform(pca_raw)

pca = PCA(n_components=2)
pca_raw = pca.fit_transform(pca_raw_scaled)

pca_high_coor_df = df[fixed_cols].copy()

pca_high_coor_df['pca_1'] = pca_raw[:, 0]
pca_high_coor_df.to_csv('correlation/filterd_by_coorelation_pca_1.csv', index=False)

pca_high_coor_df['pca_2'] = pca_raw[:, 1]
pca_high_coor_df.to_csv('correlation/filterd_by_coorelation_pca_2.csv', index=False)