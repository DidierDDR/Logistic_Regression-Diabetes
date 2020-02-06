#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import math

def standardize_Df(df):
    
    for i in range(8):
        col = df.columns[i]
        datatype = df[col].dtypes

        if (datatype == 'float64') or (datatype == 'int64'):
            std = df[col].std()
            mean = df[col].mean()
            df[col] = (df[col] - mean) / std

    return df

def apply_Function(coeficients, k):
    
    linear_function_out = 0 
    for i in range(len(coeficients)):
        if(i == 0):
            linear_function_out += coeficients[0]
        else:
            linear_function_out += k[i] * coeficients[i]
            
    return 1 / (1 + (np.power(math.e, -(linear_function_out))))


def get_Coeficients(coeficients, df):
    outcome_mean = df['Outcome'].mean()
    
    for col in range(8):
        col_mean = df[df.columns[col]].mean()
        divisor = 0
        
        for row in range(len(df.index)):
            coeficients[col+1] += (df.iloc[row, col] - col_mean) * (df.iloc[row, 8] - outcome_mean)
            divisor += np.power(df.iloc[row, col] - col_mean, 2)
        
        coeficients[col+1] /= divisor
    
    coeficients[0] = outcome_mean
    
    for i in range(8):
        coeficients[0] -=  coeficients[i+1] * df[df.columns[i]].mean()
    
    return coeficients


def get_Program_Outcome(func_out):
    if(func_out > 0.54):
        return 1
    else:
        return 0

df = pd.read_csv("https://bit.ly/2YWZ9Tn")
df = df.sort_values('Outcome')
df = df.iloc[168:]
can_not_be_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


for col in can_not_be_zero:
    df[col] = df[col].replace(0, np.NaN)
    
df = df.dropna(axis = 0, how = 'any').reset_index(drop = True)
df = standardize_Df(df)

for col in df.columns:
    df = df.loc[(df[col] < 2.5) & (df[col] > -2.5)].reset_index(drop = True)

df = df[49:].reset_index(drop = True)    
traindf = pd.concat([df.iloc[:84], df.iloc[166:249]]).sample(frac = 1).reset_index(drop = True)
testdf = pd.concat([df.iloc[84:166], df.iloc[249:]]).sample(frac = 1).reset_index(drop = True)

coeficients=[0, 0, 0, 0, 0, 0, 0, 0, 0]
coeficients = get_Coeficients(coeficients, traindf)

correct = 0 
wrong = 0 
falseneg = 0 
falsepos = 0
trueneg = 0
truepos = 0

for i in range(len(testdf.index)): 
    function_outcome = apply_Function(coeficients, testdf.loc[i])
    algorithm_outcome = get_Program_Outcome(function_outcome) 
    real_outcome = testdf.iloc[i]['Outcome']
    
    if(algorithm_outcome == real_outcome):
        if(real_outcome == 1):
            truepos += 1
        else:
            trueneg += 1
        correct += 1
    else:
        if (real_outcome == 0):
            falsepos += 1
        else:
            falseneg += 1
        wrong += 1

print('Total: '+str(correct+wrong))
print('Correct: '+str(correct))
print('Wrong: '+str(wrong)+'\n')
print('True Positives: '+str(truepos))
print('True Negatives: '+str(trueneg)+'\n')
print('Fake Positives: '+str(falsepos))
print('Fake Negatives: '+str(falseneg)+'\n')
print('Accuracy: '+str(round(100*(correct/(correct+wrong))))+'%')
print('Precision: '+str(round(100*(truepos/(truepos + falsepos))))+'%')
print('Recall: '+str(round(100*(truepos/(truepos + falseneg))))+'%')


# In[ ]:




