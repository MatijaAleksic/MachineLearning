#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Klasterovati države na osnovu njihovih karakteristika u klastere koji predstavljaju 
# geografske regione (region): Africa, Americas, Asia i Europe. Opis svih atributa je dostupan na pratećoj prezentaciji za ovaj
# zadatak. Zadatak je uspešno urađen ukoliko se na kompletnom testnom skupu podataka dobije v mera (eng. v measure score) veća
# od 0.40. Zadatak se rešava upotrebom Modela Gausovih mešavina (eng. Gaussian Mixutre Model, GMM), tj.
# algoritmom Očekivanje - maksimizacija (eng. Expectation - maximization, EM).

# Acceptance criteria: v measure > 0.4


# In[797]:


import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk

# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import v_measure_score

from random import randrange
pd.options.mode.chained_assignment = None  
import sys


# In[612]:


# df = pd.read_csv("train.csv");
# kolone = ['income', 'infant', 'region', 'oil']
#plt.scatter(df.region, df.income, color='red', marker='+')
# plt.scatter(df['income'],df['infant'])

# box = df[['income']]
# box.boxplot()

# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
 
# fig, ax = plt.subplots()

# colors = { 0.0 :'black', 1.0 :'orange', 2.0:'red', 3.0:'blue'}
# ax.scatter(df['income'], df['infant'], c=df['region'].map(colors))

# #df['region'] = df['region'].map({'Africa': 0, 'Asia': 1, 'Americas': 2, 'Europe': 3})

# plt.rcParams["figure.figsize"] = (7,2)

# europe = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',markersize=6, label='Europe')
# asia = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',markersize=6, label='Asia')
# americas = mlines.Line2D([], [], color='red', marker='o', linestyle='None',markersize=6, label='Americas')
# africa = mlines.Line2D([], [], color='black', marker='o', linestyle='None',markersize=6, label='Africa')

# plt.legend(handles=[blue_star,asia,americas,africa])

# plt.show()


# In[809]:


def fill_nan_infant(df):
    value = df['infant'].isnull()
    
    indices = []
    counter = 0
    for bol in value:
        if bol == True:
            indices.append(counter);
            #print("Index : " + str(counter) + " region : " + str(df['region'][counter]))
        counter += 1
    

    train = df
    test = pd.DataFrame()
    for i in indices:
        test = test.append(train.iloc[[i]])

    train = train.drop(labels = indices, axis=0)
    train = train.reset_index(drop=True)

    X = train.drop("infant", axis=1)
    Y = train["infant"]
    
    model = LinearRegression()
    model.fit(X, Y)

    X_test = test.drop("infant", axis=1)
    predictions = model.predict(X_test)
    
    counter1 = 0
    for index in indices:
        df['infant'][index] = predictions[counter1]
        counter1 += 1
    
    return df

def normalize_data(df):
    #std_income = np.std(df['income'])
    std_income = df['income'].std()
    #mean_income = np.mean(df['income'])
    mean_income = df['income'].mean()

    std_infant = df['infant'].std()
    mean_infant= df['infant'].mean()

    df['income'] = (df['income'] - mean_income) / std_income
    df['infant'] = (df['infant'] - mean_infant) / std_infant

    return df


# Splituje podatke na test i train odmerom split=0.7 ako se ne navede drugacije
def train_test_split_random(dataset,split=0.7):
    
    train_size_counter = 0
    indices = []
    train = pd.DataFrame()
    train_size = split * len(dataset)
    dataframe_copy = dataset
    while train_size_counter < train_size:
        train_size_counter = train_size_counter + 1
        index = randrange(len(dataset))
        
        while(check_if_element_in_list(index, indices)):
            index = randrange(len(dataset))
                 
        indices.append(index)
    
    for i in range(len(indices)+1):
        train = train.append(dataframe_copy.iloc[[i]])
        
    dataframe_copy = dataframe_copy.drop(labels = indices, axis=0)
    dataframe_copy = dataframe_copy.reset_index(drop=True)
        
    return train, dataframe_copy

def check_if_element_in_list(x, lista):
    if x in lista:
        return True
    else:
        return False
    
def encode_data(df):
    df['region'] = df['region'].map({'Africa': 0, 'Asia': 1, 'Americas': 2, 'Europe': 3})
    df['oil'] = df['oil'].map({'yes': 1, 'no': 0})
    return df


# In[826]:


# skorovi = []
# krugova = 1
# for i in range(krugova):
#     df = pd.read_csv("train.csv");
#     #kolone = ['income', 'infant', 'region', 'oil']


#     #ENKODIRANJE
#     #Rucno label enkodiranje
#     df = encode_data(df)

#     # #Enkodiranje koriscenjem LabelEncoder-a
# #     label_encoder_region = LabelEncoder()
# #     label_encoder_oil = LabelEncoder()

# #     df['region'] = label_encoder_region.fit_transform(df['region'])
# #     df['oil'] = label_encoder_oil.fit_transform(df['oil'])


    
#     #POPUNJAVANJE NAN VREDNOSTI
# #     infant_mean = df['infant'].mean()
# #     df['infant'].fillna(value=infant_mean, inplace=True)
    
#     df = fill_nan_infant(df)



#     #NORMALIZACIJA PODATAKA
#     #normalizacija income i infant kolone
#     df = normalize_data(df)

# #     ss = StandardScaler()
# #     df = ss.fit_transform(df)

#     # Da vidis da li ima NA vrednosti
#     # df.isnull().sum()

#     #Podela na train, test
#     training_data, test_data = train_test_split_random(df,split=0.75)

#     X_train = training_data.drop("region", axis=1)
#     Y_train = training_data["region"]

#     X_test = test_data.drop("region", axis=1)
#     Y_test = test_data["region"]

#     #print("train size: " +  str(len(X_train)))
#     #print("test size: " + str(len(X_test)))

#     model =GaussianMixture(n_components=4, covariance_type='diag', max_iter=2000, n_init=20, random_state=25)
#     model.fit(X_train, Y_train)
#     Y_predicted = model.predict(X_test)
#     skor = v_measure_score(Y_test, Y_predicted)
#     #print(skor)
#     skorovi.append(skor)
    
# sum = 0
# for element in skorovi:
#     #print(element)
#     sum += element

# print("Prosek " + str(krugova) + " krugova je : " + str(sum/krugova))
    

# # -4.9381, 9.35594031e-01
# # -6.16509906e-01,  3.49064846e-01


# In[ ]:


if(len(sys.argv) != 3):
    print("Mora imati dva argumenta 'train.csv' 'test.csv'")
    exit()
else:
    df_train = pd.read_csv(sys.argv[1])
    df_test = pd.read_csv(sys.argv[2])
    
    
    df_train = encode_data(df_train)
    df_test = encode_data(df_test)
    
    df_train = fill_nan_infant(df_train)

    
    #NORMALIZACIJA PODATAKA
    #normalizacija income i infant kolone
    df_train = normalize_data(df_train)
    df_test = normalize_data(df_test)


    X_train = df_train.drop("region", axis=1)
    Y_train = df_train["region"]

    X_test = df_test.drop("region", axis=1)
    Y_test = df_test["region"]

    model =GaussianMixture(n_components=4, covariance_type='diag', max_iter=2000, n_init=20, random_state=25)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    skor = v_measure_score(Y_test, Y_predicted)
    print(skor)

