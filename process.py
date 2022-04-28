import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics, preprocessing
import pickle


df = pd.read_csv('train_folds.csv')
df = pd.read_csv("train_folds.csv")
useful_features = [f for f in df.columns if f not in ['kfold', 'Outcome']]
pred_acc = [0.5]
train_acc = [0.5]
model = []
for fold in range(5):
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    x_train = df_train[useful_features]
    x_valid = df_valid[useful_features]

    y_train = df_train.Outcome
    y_valid = df_valid.Outcome

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    cat = LGBMClassifier(
        max_depth=3,
        n_estimators=700,
        random_state=42,
        learning_rate=0.01,
        verbose=0,
        reg_alpha=25,
        early_stopping_rounds=1000
    )
    cat.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
    preds_train = cat.predict(x_train)
    preds_valid = cat.predict(x_valid)
    # print(fold, metrics.accuracy_score(y_valid, preds_valid))
    train_accuracy = metrics.accuracy_score(y_train, preds_train)
    test_accuracy = metrics.accuracy_score(y_valid, preds_valid)
    train_acc.append(train_accuracy)
    pred_acc.append(test_accuracy)
    model.append(cat)
    # print(train_acc)
    # print(pred_acc)
    print(model)
print(model[1].predict([[4,120,34,43,72,12,0.77,0]]))
print(train_accuracy)
print(pred_acc)
# create an iterator object with write permission - model.pkl
with open(f'model.pkl', 'wb') as files:
    pickle.dump(model[1], files)
