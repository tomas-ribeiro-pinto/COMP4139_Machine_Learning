import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

src = 'imputed_csv/gene_predicted_rest_median_imputed.csv'

df = pd.read_csv(src)
working = df.copy()
df.drop(columns=['ID'], inplace=True)
y = df['RelapseFreeSurvival (outcome)']
X = df.drop(columns=['RelapseFreeSurvival (outcome)'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LassoCV

# Use LassoCV to find the best alpha
lasso_cv = LassoCV(cv=5).fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_
print(f"Optimal Alpha: {best_alpha}")

# Fit LASSO with the optimal alpha
lasso_optimized = Lasso(alpha=best_alpha)
lasso_optimized.fit(X_train_scaled, y_train)
y_pred_optimized = lasso_optimized.predict(X_test_scaled)

y_pred = lasso_optimized.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mae}")

# 6. Check coefficients
coef = pd.Series(lasso_optimized.coef_, index=X.columns)
print("Feature Coefficients:")
print(coef)

# Identify important features (non-zero coefficients)
important_features = coef[coef != 0].index.tolist()
print(f"Important Features: {important_features}")

'''
['HER2', 'ChemoGrade', 'TumourStage', 'Gene', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterRow', 'original_firstorder_Kurtosis', 'original_firstorder_MeanAbsoluteDeviation', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_ngtdm_Busyness']
'''


new_df = pd.read_csv(src)
important_features = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ER'] + important_features

new_df = new_df[important_features]

new_df.to_csv('lasso_csv/lasso_important_features.csv', index=False)