import pandas as pd
import numpy as np

source_csv = 'original.csv'

#1 drop rows with missing PCR
df = pd.read_csv(source_csv)
na_pcr_drapped_df = df.copy()
na_pcr_drapped_df.replace({'pCR (outcome)': {999: np.nan}}, inplace=True)
na_pcr_drapped_df.dropna(subset=['pCR (outcome)'], inplace=True)
# na_pcr_drapped_df.to_csv('imputed_csv/na_pcr_drapped.csv', index=False)

# 2. impute rest with median
median_imputed_df = na_pcr_drapped_df.copy()
median_imputed_df.replace({
  'Age':{999:np.nan},
  'ER':{999:np.nan},
  'PgR':{999:np.nan},
  'HER2':{999:np.nan},
  'TrippleNegative':{999:np.nan},
  'ChemoGrade':{999:np.nan},
  'Proliferation':{999:np.nan},
  'HistologyType':{999:np.nan},
  'LNStatus':{999:np.nan},
  'TumourStage':{999:np.nan},
  'Gene':{999:np.nan}
}, inplace=True)

IDs = median_imputed_df['ID']
median_imputed_df.drop(columns=['ID'], inplace=True)
median_imputed_df.fillna(median_imputed_df.median(), inplace=True)
median_imputed_df.insert(0, 'ID', IDs)

'''
dropped missing pcr and imputed all other columns' missing values with median
'''
median_imputed_df.to_csv('imputed_csv/na_pcr_missing_median_imputed.csv', index=False)

# 3 impute Gene with model and rest with median
gene_predicted_df = na_pcr_drapped_df.copy()
gene_predicted_df.replace({
  'Age':{999:np.nan},
  'ER':{999:np.nan},
  'PgR':{999:np.nan},
  'HER2':{999:np.nan},
  'TrippleNegative':{999:np.nan},
  'ChemoGrade':{999:np.nan},
  'Proliferation':{999:np.nan},
  'HistologyType':{999:np.nan},
  'LNStatus':{999:np.nan},
  'TumourStage':{999:np.nan},
}, inplace=True)

IDs = gene_predicted_df['ID']
gene_predicted_df.drop(columns=['ID'], inplace=True)
gene_predicted_df.fillna(gene_predicted_df.median(), inplace=True)

# prepare for model
rfs = gene_predicted_df['RelapseFreeSurvival (outcome)']
gene_predicted_df.drop(columns=[
  'RelapseFreeSurvival (outcome)'
], inplace=True)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

with open('models/missing_gene_classifier.pkl', 'rb') as f:
  clf_missing_gene = pickle.load(f)

  scaler = StandardScaler()

  target = gene_predicted_df[gene_predicted_df['Gene'] == 999]
  target = target.drop('Gene', axis=1)
  target_mri = target.iloc[:, 11:]
  target_normal_mri_data_scaled = scaler.fit_transform(target_mri)
  pca = PCA(n_components=2)
  target_pca = pca.fit_transform(target_normal_mri_data_scaled)

  target = target.iloc[:, :10]
  target['pca_1'] = target_pca[:, 0]
  target['pca_2'] = target_pca[:, 1]

  target = scaler.fit_transform(target)
  gene_pred = clf_missing_gene.predict(target)
  gene_predicted_df.loc[gene_predicted_df['Gene'] == 999, 'Gene'] = gene_pred

gene_predicted_df.insert(0, 'ID', IDs)
gene_predicted_df.insert(2, 'RelapseFreeSurvival (outcome)', rfs)
gene_predicted_df.to_csv('imputed_csv/gene_predicted_rest_median_imputed.csv', index=False)

print(f'na_pcr_drapped_df: {na_pcr_drapped_df.shape}')
print(f'median_imputed_df: {median_imputed_df.shape}')
print(f'gene_predicted_df: {gene_predicted_df.shape}')
print('done')