from sklearn.metrics import mean_absolute_error
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('imputed_csv/gene_predicted_rest_median_imputed_no_outliers.csv')

y = df['RelapseFreeSurvival (outcome)']
X = df.copy().drop(columns=['RelapseFreeSurvival (outcome)', 'ID'])

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne_mri = tsne.fit_transform(X_scale[:,11:])

# save tsne to csv
df_tsne = df.iloc[:, :11].copy()
df_tsne['tsne_1'] = X_tsne_mri[:, 0]
df_tsne['tsne_2'] = X_tsne_mri[:, 1]

df_tsne.to_csv('manifold/tsne_mri.csv', index=False)

# isomap 
isomap = Isomap(n_components=2)
X_isomap_mri = isomap.fit_transform(X_scale[:,11:])
df_isomap = df.iloc[:, :11].copy()
df_isomap['isomap_1'] = X_isomap_mri[:, 0]
df_isomap['isomap_2'] = X_isomap_mri[:, 1]

# plt.scatter(X_isomap_mri[:, -2], X_isomap_mri[:, -1], c=y, cmap="jet")
plt.scatter(X_isomap_mri[:, -2], X_isomap_mri[:, -1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()

df_isomap.to_csv('manifold/isomap_mri.csv', index=False)