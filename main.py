import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
import p
# df = pd.read_csv("Pima_Indian_Diabetes.csv")
# #I will be replacing the zeroes with nan and then filling up the nan with appropriate values
# values_with_zeroes = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
# df[values_with_zeroes] = df[values_with_zeroes].replace(0, np.NaN)
#
# #replacing nan values for insulin
# ds = df[df['Outcome']==0]
# ds['Insulin'][ds['Insulin'].notnull()].mean()
#
# #replacing nan values for insulin
# ds = df[df['Outcome']==1]
# ds['Insulin'][ds['Insulin'].notnull()].mean()
#
# df.loc[(df['Outcome']==0) & df['Insulin'].isna(), 'Insulin'] = 129.1
# df.loc[(df['Outcome']==1) & df['Insulin'].isna(), 'Insulin'] = 206.8
#
# ds = df[df['Outcome']==0]
# ds['BloodPressure'][ds['BloodPressure'].notnull()].mean()
#
# ds = df[df['Outcome']==1]
# ds['BloodPressure'][ds['BloodPressure'].notnull()].mean()
#
# #replacing the blood pressure with the mean coz replacing it with the median does not make sense
# df.loc[(df['Outcome']==0) & df['BloodPressure'].isna(), 'BloodPressure'] = 70.8
# df.loc[(df['Outcome']==1) & df['BloodPressure'].isna(), 'BloodPressure'] = 75.3
#
# df.loc[(df['Outcome']==0) & df['SkinThickness'].isna(), 'SkinThickness'] = 27.0
# df.loc[(df['Outcome']==1) & df['SkinThickness'].isna(), 'SkinThickness'] = 32.0
#
# #replacing the glucose values mean bcoz it seems appropriate to do so
# df.loc[(df['Outcome']==0) & df['Glucose'].isna(), 'Glucose'] = 107.0
# df.loc[(df['Outcome']==1) & df['Glucose'].isna(), 'Glucose'] = 140.0
#
# df.loc[(df['Outcome']==0) & df['BMI'].isna(), 'BMI'] = 30.8
# df.loc[(df['Outcome']==1) & df['BMI'].isna(), 'BMI'] = 35.4
#
# df['Pregnancies'] = df['Pregnancies'].fillna(4.0)
# df['Pregnancies'] = df['Pregnancies'].astype('int')
#
# df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].fillna(df['DiabetesPedigreeFunction'].mean())
#
# df['kfold'] = -1
# y = df.Outcome
# df = df.sample(frac=1).reset_index(drop=True)
# kf = model_selection.StratifiedKFold(n_splits=5)
# for fold, (trn_, vld_) in enumerate(kf.split(X=df, y=y)):
#     df.loc[vld_, 'kfold'] = fold
# df.to_csv("train_folds.csv", index=False)