missing_gene_classifier
- input has to be as follow

`target = X[X['Gene'] == 999]
target = target.drop('Gene', axis=1)
target_mri = target.iloc[:, 11:]
target_normal_mri_data_scaled = scaler.fit_transform(target_mri)
pca = PCA(n_components=2)
target_pca = pca.fit_transform(target_normal_mri_data_scaled)

target = target.iloc[:, :10]
target['pca_1'] = target_pca[:, 0]
target['pca_2'] = target_pca[:, 1]

target = scaler.fit_transform(target)
gene_pred = clf_missing_gene.predict(target)`