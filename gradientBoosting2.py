from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import numpy as np
import pandas as pd

def gradientBoosting(state=42):
    gradientBoostingClassifier = GradientBoostingClassifier(
        loss='deviance', 
        learning_rate=0.1, 
        n_estimators=100, 
        # subsample=1.0, 
        subsample=0.6, #// 0.5 < x <0.8
        criterion='friedman_mse', 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_depth=3, 
        # max_depth=6, #// 4 < x < 8
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        init=None, 
        random_state=state, 
        max_features=None, 
        # verbose=0,
        verbose=1, # 1 para saida das interações
        max_leaf_nodes=None, 
        warm_start=False, 
        presort='auto'
    )
    return (gradientBoostingClassifier, "Gradient Boosting") 

def printar(classifier, answerAll=42, name = "Classificador"):
    
    print(f'rodado: {name}')

    np.random.seed(answerAll)

    data_set = pd.read_csv('data/TRN',sep='\t')
    data_set.drop_duplicates(inplace=True)  # Remove exemplos repetidos

    # Também convertemos os dados para arrays ao invés de DataFrames
    X = data_set.iloc[:, :-2].values
    y = data_set.iloc[:, -1].values

    # Treino: 50%, Validação: 25%, Teste: 25%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                        random_state=answerAll, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
                                                    random_state=answerAll, stratify=y_train)

    fit = classifier.fit(X_train, y_train)
    # joblib.dump(classifier, "gradientBoosting1.txt")

    score = classifier.score(X_test, y_test)
    print(f'O Score do {name} é {score}')

    accuracy = cross_val_score(classifier, X_test, y_test, scoring='accuracy')
    average_precision = cross_val_score(classifier, X_test, y_test, scoring='average_precision')
    precision = cross_val_score(classifier, X_test, y_test, scoring='precision')
    recall = cross_val_score(classifier, X_test, y_test, scoring='recall')
    roc_auc = cross_val_score(classifier, X_test, y_test, scoring='roc_auc')

    print(f'Accuracy: {accuracy}')
    print(f'average_precision: {average_precision}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'roc_auc: {roc_auc}')

def main():
    answerAll = 42
    classifier, name = gradientBoosting(answerAll)
    printar(classifier, answerAll, name)

if __name__ == "__main__":
    main()