#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor


# In[2]:


def train_test_split(data):
    Input = pd.DataFrame()
    Output = pd.DataFrame()

    Input['test_case_type'] = data['test_case_type']
    Input['max_grid_points'] = data['max_grid_points']
    Input['newton_critical_tolerance'] = np.log(data['newton_critical_tolerance'])
    Input['newton_armijo_probes'] = data['newton_armijo_probes']
    Input['newton_max_iterations'] = data['newton_max_iterations']
    Input['newton_tolerance'] = np.log(data['newton_tolerance'])
    Input['add_factor'] = data['add_factor']
    Input['remove_factor'] = data['remove_factor']
    Input['use_collocation_scaling'] = data['use_collocation_scaling']

    Output['nODEevals'] = data['nODEevals']
    Output['nGridPoints'] = data['nGridPoints']
    Output['maxResiduum'] = data['maxResiduum']

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Input, Output, train_size = 0.8, 
                                                                                test_size = 0.2, random_state=101)
    
    #normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def getModelsDict():
    #lgbm regressor
    
    # hyper parameters after tuning
    new_params = {
    'n_estimators':874, 
    'learning_rate':0.0954093125341308, 
    'min_child_samples':15, 
    'num_leaves':50
    }
    
    rgr_lgbm = MultiOutputRegressor(lgb.LGBMRegressor(**new_params), n_jobs=-1)

    # randon forest regressor
    rgr_rf = RandomForestRegressor(max_depth=2, random_state=0)

    #knn regressor
    rgr_knn = MultiOutputRegressor(KNeighborsRegressor(), n_jobs=-1)

    #artificial neural net regressor

    #hyper parameters
    hidden_units=250
    learning_rate=0.01
    hidden_layer_act='relu'
    output_layer_act='linear'

    rgr_ann = Sequential()

    rgr_ann.add(Dense(hidden_units, input_dim=9, activation=hidden_layer_act))
    rgr_ann.add(Dense(hidden_units, activation=hidden_layer_act))
    rgr_ann.add(Dense(hidden_units, activation=hidden_layer_act))
    rgr_ann.add(Dense(hidden_units, activation=hidden_layer_act))
    rgr_ann.add(Dense(hidden_units, activation=hidden_layer_act))
    rgr_ann.add(Dense(hidden_units, activation=hidden_layer_act))
    rgr_ann.add(Dense(3, activation=output_layer_act))

    sgd = optimizers.SGD(lr=learning_rate)
    rgr_ann.compile(loss='mean_absolute_error',optimizer=sgd, metrics=['mean_absolute_error'])
    
    model_dict = {'LGBM_Rgr': rgr_lgbm, 'KNN_Rgr': rgr_knn, 'ANN_Rgr': rgr_ann, 'RF_Rgr': rgr_rf}
    
    return model_dict


# In[3]:


def main():
    data = pd.read_csv('RegressionDataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(data)
    model_dict = getModelsDict()
    
    #tranform the outputs to have normal distribution
    transformer = QuantileTransformer(output_distribution='normal')

    for model in model_dict:
        print(model)
        if model == 'ANN_Rgr':
            #explicitely transform the output distribution for neural n
            quantile = QuantileTransformer(output_distribution='normal')
            y_train_normal = quantile.fit_transform(y_train)
            model_dict[model].fit(X_train, y_train_normal, epochs=10, batch_size=32,  verbose=2)

            model_dict[model].save('ANN_Rgr')
            y_pred = model_dict[model].predict(X_test)
            # inverse tranformation to the original distribution
            y_pred = quantile.inverse_transform(y_pred)
        else:
            regr = TransformedTargetRegressor(regressor=model_dict[model],
                                  transformer=transformer)
            regr.fit(X_train, y_train)
            pickle.dump(regr, open(model, 'wb'))
            
            y_pred = regr.predict(X_test)
        
        print("Coefficient of determination ", r2_score(y_test, y_pred))
        print("Root Mean Squared Error ", mean_squared_error(y_test, y_pred, squared=False))
        print('Mean Absolute Percentage Error ', mean_absolute_percentage_error(y_test, y_pred))
        
        print('__________________________________________________')


# In[ ]:


if __name__=='__main__':
    main()


# In[ ]:




