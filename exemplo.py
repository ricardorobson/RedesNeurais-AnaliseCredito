from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC

import numpy as np
import pandas as pd

answerAll = 42

np.random.seed(answerAll)

data_set = pd.read_csv('data/TRN',sep='\t')
data_set.drop_duplicates(inplace=True)  # Remove exemplos repetidos

# Também convertemos os dados para arrays ao invés de DataFrames
X = data_set.iloc[:, :-2].values
y = data_set.iloc[:, -1].values

## Treino: 50%, Validação: 25%, Teste: 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                    random_state=answerAll, stratify=y)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
#                                                 random_state=answerAll, stratify=y_train)



# mlpClassifier = MLPClassifier(
#     hidden_layer_sizes=5,
#     activation='tanh',
#     batch_size=1000,
#     learning_rate='adaptive',
#     max_iter=10,
#     random_state=answerAll,
#     # verbose=True,
#     early_stopping=True,
#     tol=10,
# ) 

# fit = mlpClassifier.fit(X_train, y_train)
# score = mlpClassifier.score(X_test,y_test)
# print(f'O Score do MLP é {score}')

# accuracy = cross_val_score(mlpClassifier, X_test, y_test, scoring='accuracy')
# average_precision = cross_val_score(mlpClassifier, X_test, y_test, scoring='average_precision')
# precision = cross_val_score(mlpClassifier, X_test, y_test, scoring='precision')
# recall = cross_val_score(mlpClassifier, X_test, y_test, scoring='recall')
# roc_auc = cross_val_score(mlpClassifier, X_test, y_test, scoring='roc_auc')

# print(f'Accuracy: {accuracy}')
# print(f'average_precision: {average_precision}')
# print(f'precision: {precision}')
# print(f'recall: {recall}')
# print(f'roc_auc: {roc_auc}')


# gradientBoostingClassifier = GradientBoostingClassifier(
#     loss='deviance', 
#     learning_rate=0.1, 
#     n_estimators=100, 
#     subsample=1.0, 
#     criterion='friedman_mse', 
#     min_samples_split=2, 
#     min_samples_leaf=1, 
#     min_weight_fraction_leaf=0.0, 
#     max_depth=3, 
#     min_impurity_decrease=0.0, 
#     min_impurity_split=None, 
#     init=None, 
#     random_state=None, 
#     max_features=None, 
#     verbose=0, 
#     max_leaf_nodes=None, 
#     warm_start=False, 
#     presort='auto'
# )

# fit = gradientBoostingClassifier.fit(X_train, y_train)
# score = gradientBoostingClassifier.score(X_test,y_test)
# print(f'O Score do GradientBoosting é {score}')

# accuracy = cross_val_score(gradientBoostingClassifier, X_test, y_test, scoring='accuracy')
# average_precision = cross_val_score(gradientBoostingClassifier, X_test, y_test, scoring='average_precision')
# precision = cross_val_score(gradientBoostingClassifier, X_test, y_test, scoring='precision')
# recall = cross_val_score(gradientBoostingClassifier, X_test, y_test, scoring='recall')
# roc_auc = cross_val_score(gradientBoostingClassifier, X_test, y_test, scoring='roc_auc')

# print(f'Accuracy: {accuracy}')
# print(f'average_precision: {average_precision}')
# print(f'precision: {precision}')
# print(f'recall: {recall}')
# print(f'roc_auc: {roc_auc}')


# randomForestClassifier = RandomForestClassifier(
#     n_estimators=10, 
#     criterion='gini', 
#     max_depth=None, 
#     min_samples_split=2, 
#     min_samples_leaf=1, 
#     min_weight_fraction_leaf=0.0, 
#     max_features='auto', 
#     max_leaf_nodes=None, 
#     min_impurity_decrease=0.0, 
#     min_impurity_split=None, 
#     bootstrap=True, 
#     oob_score=False, 
#     n_jobs=1, 
#     random_state=42, 
#     verbose=0, 
#     warm_start=False, 
#     class_weight=None
# )

# fit = randomForestClassifier.fit(X_train, y_train)
# score = randomForestClassifier.score(X_test,y_test)
# print(f'O Score do RandomForestClassifier é {score}')

# accuracy = cross_val_score(randomForestClassifier, X_test, y_test, scoring='accuracy')
# average_precision = cross_val_score(randomForestClassifier, X_test, y_test, scoring='average_precision')
# precision = cross_val_score(randomForestClassifier, X_test, y_test, scoring='precision')
# recall = cross_val_score(randomForestClassifier, X_test, y_test, scoring='recall')
# roc_auc = cross_val_score(randomForestClassifier, X_test, y_test, scoring='roc_auc')

# print(f'Accuracy: {accuracy}')
# print(f'average_precision: {average_precision}')
# print(f'precision: {precision}')
# print(f'recall: {recall}')
# print(f'roc_auc: {roc_auc}')

linearSVC = LinearSVC(
    penalty='l2', 
    loss='squared_hinge', 
    dual=True, 
    tol=0.0001, 
    C=1.0, 
    multi_class='ovr', 
    fit_intercept=True, 
    intercept_scaling=1, 
    class_weight=None, 
    verbose=0, 
    random_state=None, 
    max_iter=1000
)

fit = linearSVC.fit(X_train, y_train)
score = linearSVC.score(X_test,y_test)
print(f'O Score do LinearSVC é {score}')

accuracy = cross_val_score(linearSVC, X_test, y_test, scoring='accuracy')
average_precision = cross_val_score(linearSVC, X_test, y_test, scoring='average_precision')
precision = cross_val_score(linearSVC, X_test, y_test, scoring='precision')
recall = cross_val_score(linearSVC, X_test, y_test, scoring='recall')
roc_auc = cross_val_score(linearSVC, X_test, y_test, scoring='roc_auc')

print(f'Accuracy: {accuracy}')
print(f'average_precision: {average_precision}')
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'roc_auc: {roc_auc}')