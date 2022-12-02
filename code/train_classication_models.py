#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pickle


# In[3]:


def train_test_split(data):
    Input = pd.DataFrame()
    Output = pd.DataFrame()

    Input['max_grid_points'] = data['max_grid_points']
    Input['newton_critical_tolerance'] = np.log(data['newton_critical_tolerance'])
    Input['newton_armijo_probes'] = data['newton_armijo_probes']
    Input['newton_max_iterations'] = data['newton_max_iterations']
    Input['newton_tolerance'] = np.log(data['newton_tolerance'])
    Input['add_factor'] = data['add_factor']
    Input['remove_factor'] = data['remove_factor']
    Input['use_collocation_scaling'] = data['use_collocation_scaling']

    Output['success'] = data['success']

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Input, Output, train_size = 0.8, 
                                                                                test_size = 0.2, random_state=101)
    
    #normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def getModelsDict():
    #lgbm classifier
    clf_lgbm = lgb.LGBMClassifier()

    # randon forest classifier
    clf_rf = RandomForestClassifier(max_depth=2)

    #knn classifier
    clf_knn = KNeighborsClassifier()

    #artificial neural net classifier

    #hyper parameters
    hidden_units=100
    learning_rate=0.01
    hidden_layer_act='tanh'
    output_layer_act='sigmoid'

    clf_ann = Sequential()
    clf_ann.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
    clf_ann.add(Dense(hidden_units, activation=hidden_layer_act))
    clf_ann.add(Dense(1, activation=output_layer_act))

    sgd=optimizers.SGD(lr=learning_rate)
    clf_ann.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])
    
    model_dict = {'LGBM_Clf': clf_lgbm, 'KNN_Clf': clf_knn, 'ANN_Clf': clf_ann, 'RF_Clf': clf_rf}
    
    return model_dict


# In[4]:


def main():
    data = pd.read_csv('ClassificationDataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(data)
    y_train = y_train['success'].to_numpy().tolist()
    y_test = y_test['success'].to_numpy().tolist()
    model_dict = getModelsDict()

    for model in model_dict:
        print(model)
        if model == 'ANN_Clf':
            model_dict[model].fit(X_train, y_train, epochs=100, batch_size=64,  verbose=2)
            model_dict[model].save('ANN_Clf')
            y_pred = model_dict[model].predict(X_test)
            y_pred = (y_pred > 0.5).astype(int).flatten()
        else: 
            model_dict[model].fit(X_train, y_train)
            pickle.dump(model_dict[model], open(model, 'wb'))
            y_pred = model_dict[model].predict(X_test)
        
        precision = precision_score(y_test, y_pred, average='binary')
        print("Precision: ", precision)
        
        recall = recall_score(y_test, y_pred, average='binary')
        print("Recall: ", recall)
        
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        
        print("Confusion Matrix ")
        print(confusion_matrix(y_test, y_pred))
        print('__________________________________________________')


# In[5]:


if __name__=='__main__':
    main()


# In[ ]:




