from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE, ADASYN

import os.path as path
import numpy as np
import pandas as pd
import timeit

def gradientBoosting(state=42):
    gradientBoostingClassifier = GradientBoostingClassifier(
        loss='deviance', 
        learning_rate=0.1, 
        n_estimators=100, 
        subsample=1.0, 
        # subsample=0.6, // 0.5 < x <0.8
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
    return (gradientBoostingClassifier, "Gradient_Boosting_1") 

def dividir(answerAll=42):
    np.random.seed(answerAll)

    data_set = pd.read_csv('data/TRN',sep='\t')
    data_set.drop_duplicates(inplace=True)  # Remove exemplos repetidos

    # Também convertemos os dados para arrays ao invés de DataFrames
    X = data_set.iloc[:, :-3].values
    y = data_set.iloc[:, -2].values
 
    # Treino: 50%, Validação: 25%, Teste: 25%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                        random_state=answerAll, stratify=y)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
    #                                                 random_state=answerAll, stratify=y_train)

    # train_test_split(y, shuffle=False)

    X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X_train, y_train)

    return (X_resampled, y_resampled, X_test, y_test)


def metricas(classifier, X_test, y_test, name = "Classificador"):
    
    score = classifier.score(X_test, y_test)
    accuracy = cross_val_score(classifier, X_test, y_test, scoring='accuracy')
    average_precision = cross_val_score(classifier, X_test, y_test, scoring='average_precision')
    precision = cross_val_score(classifier, X_test, y_test, scoring='precision')
    recall = cross_val_score(classifier, X_test, y_test, scoring='recall')
    roc_auc = cross_val_score(classifier, X_test, y_test, scoring='roc_auc')
  
    score = f'Score: {score}'
    accuracy = f'Accuracy: {accuracy}'
    average_precision = f'average_precision: {average_precision}'
    precision = f'precision: {precision}'
    recall = f'recall: {recall}'
    roc_auc = f'roc_auc: {roc_auc}'

    return [score, accuracy, average_precision, precision, recall, roc_auc]


def saveLog(classifier, metricas, file_name, time, name):
    
    name = "log/" + name + ".txt"
    texto = []
    try:
        arquivo = open(name, 'r')
        texto = arquivo.readlines()
        arquivo = open(name, 'w')
    except FileNotFoundError:
        arquivo = open(name, 'w')
   
    texto.append("Rede treinada:\n")
    texto.append(file_name)
    texto.append("\n\nTempo de treinamento:\n")
    texto.append("{0:.2f}s".format(time))
    texto.append("\n\nConfiguração do classificador:\n")
    texto.append(classifier.__str__())
    texto.append("\n\nMetricas do classificador:\n")
   
    for metrica in metricas:
        texto.append(metrica+"\n")

    texto.append("---------------------------------------------------------\n\n")

    # escrever no arquivo
    arquivo.writelines(texto)
    arquivo.close()

def saveModel(classifier, name = "Classificador"):

    name = "log/" + name
    i = 0
    while path.isfile(name + '_' + str(i) + '.pkl'):
        i = i + 1

    name_new = name + "_" + str(i) + ".pkl"
    joblib.dump(classifier, name_new)
    return name_new

def main():

    answerAll = 42
    start = timeit.default_timer()

    # divide os dados de entrada
    X_train, y_train, X_test, y_test = dividir(answerAll)

    loadModel = False

    if loadModel:
         # classificadores salvos
        name = "Gradient_Boosting_1"
        path = "log/Gradient_Boosting_1_0.pkl"
        classifier = joblib.load(path) 

        metricas_list = metricas(classifier, X_test, y_test, name)
        print("Metricas do classificador: " + name)
        for metrica in metricas_list:
            print(metrica)

        stop = timeit.default_timer()
        print("Tempo de execução: {0:.2f}s".format(stop - start))
    else:
        classifier, name = gradientBoosting(answerAll)
        
        # treinar o modelo
        classifier.fit(X_train, y_train)

        # printar os resultados das metricas
        metricas_list = metricas(classifier, X_test, y_test, name)
        print("Metricas do classificador: " + name)
        for metrica in metricas_list:
            print(metrica)

        name_model = saveModel(classifier, name)
        stop = timeit.default_timer()
        time = stop - start
        print("Tempo de execução: {0:.2f}s".format(time))
        saveLog(classifier, metricas_list, name_model, time, name)


if __name__ == "__main__":
    main()