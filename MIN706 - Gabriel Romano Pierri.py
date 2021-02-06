"""
    
    1) CONTEXTUALIZAÇÃO

	Apesar do agro gerar muito lucro, a vida dos agricultores não é fácil, mas sim um verdadeiro teste de resistência e determinação. Uma vez que tenhamos semeado as sementes, o agricultor precisa trabalhar dia e noite para garantir uma boa safra no final da estação. 
    
    Uma boa colheita depende de diversos fatores, como disponibilidade de água, fertilidade do solo, proteção das culturas, uso oportuno de pesticidas, outros fatores químicos úteis e da natureza.

​	Muitos desses dados são quase impossíveis de se controlar, mas a quantidade e a frequência de pesticidas é algo que o agricultor pode administrar. Os pesticidas podem protegem a colheita com a dosagem certa. Mas, se adicionados em quantidade inadequada, podem prejudicar toda a safra.

    Dito isto, são fornecidos dados baseados em culturas colhidas por vários agricultores no final da safra de 2018-2019. Para simplificar o problema, é assumido que todos os fatores relacionados as técnicas agrícolas e climáticas, não influenciaram esses resultados.


​	2) OBJETIVO 
    
​	O objetivo neste projeto é determinar o resultado da safra atual de 2020, ou seja, se a colheita será saudável, prejudicada por pesticidas, ou prejudicada por outros motivos.


   3) DESCRIÇÃO DAS VARIÁVEIS

    | ------------------------ | ------------------------------------------------------------ |
    | Variável                 | Descrição                                                    |
    | ------------------------ | ------------------------------------------------------------ |
    | Identificador_Agricultor | IDENTIFICADOR DO CLIENTE                                     |
    | Estimativa_de_Insetos    | Estimativa de insetos por M²                                 |
    | Tipo_de_Cultivo          | Classificação do tipo de cultivo (0,1)                       |
    | Tipo_de_Solo             | Classificação do tipo de solo (0,1)                          |
    | Categoria_Pesticida      | Informação do uso de pesticidas (1- Nunca Usou, 2-Já Usou, 3-Esta usando) |
    | Doses_Semana             | Número de doses por semana                                   |
    | Semanas_Utilizando       | Número de semanas Utilizada                                  |
    | Semanas_Sem_Uso          | Número de semanas sem utilizar                               |
    | Temporada                | Temporada Climática (1,2,3)                                  |
    | dano_na_plantacao        | Variável de Predição - Dano no Cultivo (0=Sem Danos, 1=Danos causados por outros motivos, 2=Danos gerados pelos pesticidas) |
    | ------------------------ | ------------------------------------------------------------ |

"""



####################################################################################################
#                                        IMPORTANDO BIBLIOTECAS                                    #
####################################################################################################

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r'C:\Users\gabri\OneDrive\Documentos\Minor - Ciência de Dados\MIN706 - Projetos em Ciência de Dados\TCC - MINOR\Safra_2018-2019.csv', encoding = 'Latin1', sep = ';')
df_2020 = pd.read_csv(r'C:\Users\gabri\OneDrive\Documentos\Minor - Ciência de Dados\MIN706 - Projetos em Ciência de Dados\TCC - MINOR\Safra_2020.csv', encoding = 'Latin1', sep = ';')


#df = pd.read_csv(r'C:\Users\gusta\Documents\modelo_safra_imt\Safra_2018-2019.csv', encoding = 'Latin1', sep = ';')
#df_2020 = pd.read_csv(r'C:\Users\gusta\Documents\modelo_safra_imt\Safra_2020.csv', encoding = 'Latin1', sep = ';')

####################################################################################################
#                                         DATA PREPARATION                                         #
####################################################################################################

"""

1º) ALTERAÇÃO DE VALORES CATEGÓRICOS PARA NUMÉRICAS E EXCLUSÃO DA COLUNA DE AGRICULTORES

"""
df = df.drop(columns=['Identificador_Agricultor'])

df['dano_na_plantacao'] = df['dano_na_plantacao'].replace('Sem Danos',0).replace('Danos causados por outros motivos',1).replace('Danos gerados pelos pesticidas',2)

df['Categoria_Pesticida'] = df['Categoria_Pesticida'].replace('Nunca Usou',1).replace('Já Usou',2).replace('Esta usando',3)

df.groupby(by = 'dano_na_plantacao')['dano_na_plantacao'].count()

"""

2º) TRATAMENTO DE VALORES MISSINGS/NULOS

RESULTADO: O MELHOR MÉTODO FOI O DE INTERPOLAÇÃO, VISTO QUE SUA MÉDIA DEU MUITO SIMILAR COM POUCA VARIAÇÃO
AO DATAFRAME NÃO TRATADO, POSSUÍU O VALOR DA SUA MEDIANA IGUAL E O VALOR DE CORRELAÇÃO FOI A QUE MENOS VARIOU.
           

"""


df.isnull().sum()

#Identificador_Agricultor       0
#Estimativa_de_Insetos          0
#Tipo_de_Cultivo                0
#Tipo_de_Solo                   0
#Categoria_Pesticida            0
#Doses_Semana                   0
#Semanas_Utilizando          8055 - 10% DA BASE
#Semanas_Sem_Uso                0
#Temporada                      0
#dano_na_plantacao              0

print('NÚMERO DE VALORES NULOS TRATADOS:',df.isnull().sum()[df.isnull().sum() > 0][0])


MEAN = []
MEDIAN = []
CORR_TARGET = []

df.corr().abs()['dano_na_plantacao']

na = df['Semanas_Utilizando'].describe()[1]
MEAN.append(na) 
na = df['Semanas_Utilizando'].describe()[5]
MEDIAN.append(na) 
na = df.corr()['Semanas_Utilizando'][8]
CORR_TARGET.append(na)   
   
fill_mean = df['Semanas_Utilizando'].fillna(df['Semanas_Utilizando'].describe()[1]).describe()[1]
MEAN.append(fill_mean)
fill_mean = df['Semanas_Utilizando'].fillna(df['Semanas_Utilizando'].describe()[1]).describe()[5]
MEDIAN.append(fill_mean) 
fill_mean = df.fillna(df['Semanas_Utilizando'].describe()[1]).corr()['Semanas_Utilizando'][8]
CORR_TARGET.append(fill_mean)  

fill_median = df['Semanas_Utilizando'].fillna(df['Semanas_Utilizando'].describe()[5]).describe()[1]
MEAN.append(fill_median)
fill_median = df['Semanas_Utilizando'].fillna(df['Semanas_Utilizando'].describe()[5]).describe()[5]
MEDIAN.append(fill_median) 
fill_median = df.fillna(df['Semanas_Utilizando'].describe()[5]).corr()['Semanas_Utilizando'][8]
CORR_TARGET.append(fill_median)  

fill_zero = df['Semanas_Utilizando'].fillna(0).describe()[1]
MEAN.append(fill_zero)
fill_zero = df['Semanas_Utilizando'].fillna(0).describe()[5]
MEDIAN.append(fill_zero) 
fill_zero = df.fillna(0).corr()['Semanas_Utilizando'][8]
CORR_TARGET.append(fill_zero)

interpolate = df['Semanas_Utilizando'].interpolate().describe()[1]
MEAN.append(interpolate)
interpolate = df['Semanas_Utilizando'].interpolate().describe()[5]
MEDIAN.append(interpolate) 
interpolate = df.interpolate().corr()['Semanas_Utilizando'][8]
CORR_TARGET.append(interpolate)

dropna = df['Semanas_Utilizando'].dropna().describe()[1]
MEAN.append(dropna)
dropna = df['Semanas_Utilizando'].dropna().describe()[5]
MEDIAN.append(dropna) 
dropna = df.dropna().corr()['Semanas_Utilizando'][8]
CORR_TARGET.append(dropna)


df_comparate = pd.DataFrame({'MÉDIA':MEAN,'MEDIANA':MEDIAN,'CORR_TARGET':CORR_TARGET}, 
                            index = ['FILLNA(NA)','FILLNA(MEAN)','FILLNA(MEDIAN)','FILLNA(0)','INTERPOLATE()','DROPNA()'])

df_comparate

df['Semanas_Utilizando'] = df['Semanas_Utilizando'].interpolate().astype('int64')


"""

3º) IDENTIFICAÇÃO E TRATAMENTO DE OUTLIERS

           
"""

# CONTANDO NÚMERO DE OUTLIERS

COLUMNS = ['Estimativa_de_Insetos','Tipo_de_Cultivo','Tipo_de_Solo','Doses_Semana','Semanas_Utilizando','Semanas_Sem_Uso','Temporada']

OUTLIERS = 0

for col in COLUMNS:
    print(col)
    Q3 = df[col].quantile(0.75)
    Q1 = df[col].quantile(0.25)
    LSE = Q3 + 1.5*(Q3 - Q1)
    LIE = Q1 - 1.5*(Q3 - Q1)    
    OUTLIERS = OUTLIERS +  df[df[col] > LSE][col].count() + df[df[col] < LIE][col].count()
    print(df[df[col] > LSE][col].count() + df[df[col] < LIE][col].count())

print(OUTLIERS,'OUTLIERS')

# EXCLUSÃO DOS OUTLIERS

for col in COLUMNS:
    print(col)
    Q3 = df[col].quantile(0.75)
    Q1 = df[col].quantile(0.25)
    LSE = Q3 + 1.5*(Q3 - Q1)
    LIE = Q1 - 1.5*(Q3 - Q1)  
    if (df[df[col] > LSE][col].count() + df[df[col] < LIE][col].count()) > 0:
        df = df[df[col] <= LSE]

print('NÚMERO EM PORCENTAGEM DA BASE EXCLUÍDA:',(1 - df[col].count() / 80000) * 100,'%')


"""

4º) LIMPEZA DE VARIÁVEIS CONSTANTES 

EXEMPLO: 99% DE VARIÁVEIS PREENCHIDAS COM 0 OU NULL DEVEM SER EXCLUÍDAS UMA VEZ QUE NÃO SÃO SIGNIFICATIVAS NO MODELO
              
"""

from sklearn.feature_selection import VarianceThreshold

var_thres = VarianceThreshold(threshold = 0.01)
var_thres.fit(df)
var_thres.get_support()

constant_columns = [column for column in df.columns
                    if column not in df.columns[var_thres.get_support()]]


df.drop(constant_columns, axis = 1)

print('NÚMERO DE VARIÁVEIS CONSTANTES EXCLUÍDAS:',len(constant_columns))


"""

5º) SELEÇÃO DAS MELHORES VARIÁVEIS PARA O MODELO

              
"""

df.corr().abs()['dano_na_plantacao'].sort_values(ascending = False)


# CRIAÇÃO DE UMA COLUNA ALEATÓRIA PARA UTILIZAR DE PARÂMETRO DE CORTE 

df['RANDOM'] = np.random.randint(1,10,77665)


# SELECÃO DO PARÂMETRO CORTE E POSTERIOR EXCLUSÃO DA(S) COLUNA(S) NO DF INICIAL

df_corr_random = df.corr().abs()['dano_na_plantacao'].sort_values(ascending = False).reset_index(drop = False)
num_random = df_corr_random[df_corr_random['index'] == 'RANDOM']['dano_na_plantacao'].reset_index(drop = True)[0]
columns = df_corr_random[df_corr_random['dano_na_plantacao'] > num_random]['index'].to_list()

for col in df.columns:
    if col not in columns:
        df = df.drop(col, axis = 1)

print('NÚMERO DE VARIÁVEIS ALEATÓRIAS EXCLUÍDAS: ',df_corr_random['index'].count() - len(columns) - 1)



"""

6º) SEPARANDO DATAFRAME EM TREINO E TESTE

              
"""

from sklearn import model_selection

X = df.drop(columns='dano_na_plantacao') #ATRIBUTO DE ENTRADA
y = df['dano_na_plantacao']   #ATRIBUTO DE SAÍDA

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)


"""

7º) LIMPEZA DE VARIÁVEIS CORRELACIONADAS

              
"""

corr = df.corr()
corr = corr.abs().unstack()
corr = corr.sort_values(ascending=False)
corr = corr[corr >= 0.4]
corr = corr[corr < 1]
corr = pd.DataFrame(corr).reset_index()
corr.columns = ['var1', 'var2', 'corr']

grupo_vars = []
correlacao_grupo = []

for variavel in corr.var1.unique():
    if variavel not in grupo_vars:
        bloco_correl = corr[corr.var1 == variavel]
        grupo_vars = grupo_vars + list(bloco_correl.var2.unique()) + [variavel]

        correlacao_grupo.append(bloco_correl) 
    
vars_correl_drop = []
from sklearn.ensemble import RandomForestClassifier
for grupo in correlacao_grupo:
    variaveis = list(grupo.var2.unique()) + list(grupo.var1.unique())
    rf = RandomForestClassifier(n_estimators = 200, max_depth=4, random_state=0, n_jobs=-1)
    rf.fit(X_train[variaveis], y_train)
    rf.feature_importances_
    
importancia_vars = pd.DataFrame(rf.feature_importances_, index = X_train[variaveis].columns, columns = ['importance']).sort_values('importance',ascending=False)


"""

8º) PREPROCESSING - NORMALIZAÇÃO DOS DADOS

              
"""
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

cols = X_train.columns.to_list()

sca = preprocessing.StandardScaler()
X_train = sca.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=cols)
X_test = sca.transform(X_test)
X_test = pd.DataFrame(X_test, columns=cols)

"""

9º) MACHINE LEARNING

              
"""
#!pip install xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

######################################################################

X_train.shape
X_test.shape
X = pd.concat([X_test, X_train])
y = pd.concat([y_test, y_train])

modelos = [RandomForestClassifier,
           DecisionTreeClassifier,
           GaussianNB,
           KNeighborsClassifier,
           xgboost.XGBClassifier,
           DummyClassifier,
           GradientBoostingClassifier,
           SVC
           ]

kfold = KFold(n_splits=10, shuffle=True, random_state=0)
rank = []

for model in modelos:
    print(model)
    clf = model()
    scores = cross_val_score(clf, X, y, cv=kfold)
    rank.append(scores.mean())
    print(rank[-1])

df_rank = pd.DataFrame({'modelo':modelos,'scores':rank}, columns=['modelo','scores'])
df_rank.sort_values(by='scores',ascending=False)

########################################################################

#XGBClassifier:
xgb_par = {'n_estimators': list(range(300,550,100)),
            'objective':['binary:logistic'] ,
            'min_child_weight': [4,5],
            'gamma': [0.5,1],
            'subsample': [0.6,1.0],
            'colsample_bytree': [0.8,1.0],
            'max_depth': [3,4,5] 
            }

clf_xgb = GridSearchCV(xgboost.XGBClassifier(), xgb_par, cv=5, n_jobs=-1)
clf_xgb.fit(X_train,y_train)

clf_xgb_1 = xgboost.XGBClassifier(
n_estimators = clf_xgb.best_params_['n_estimators'],
objective = clf_xgb.best_params_['objective'],
min_child_weight = clf_xgb.best_params_['min_child_weight'],
gamma = clf_xgb.best_params_['gamma'],
subsample = clf_xgb.best_params_['subsample'],
colsample_bytree = clf_xgb.best_params_['colsample_bytree'],
max_depth = clf_xgb.best_params_['max_depth'])

y_pred_test = clf_xgb_1.predict(X_test) 
np.mean(y_pred_test  == y_test) # clf_rf_2.score(X_test,y_test)

y_pred_train = clf_xgb_1.predict(X_train) 
np.mean(y_pred_train == y_train) # clf_rf_2.score(X_train,y_train)

#GradientBoostingClassifier:
gbc_par = {'learning_rate': [0.01],
            'n_estimators': [300, 400, 500],
            'max_depth': [4,5],
            'min_samples_split': list(range(200,550,150)),
            'min_samples_leaf': list(range(50,200,50)),
            'max_features':['sqrt', 'log2'],
            'criterion':['friedman_mse']
            }

clf_gbc = GridSearchCV(GradientBoostingClassifier(), gbc_par, cv=5, n_jobs=-1)
clf_gbc.fit(X_train,y_train)

clf_gbc_1 = GradientBoostingClassifier(
learning_rate = clf_gbc.best_params_['learning_rate'],
n_estimators = clf_gbc.best_params_['n_estimators'],
max_depth = clf_gbc.best_params_['max_depth'],
loss = clf_gbc.best_params_['loss'],
min_samples_split = clf_gbc.best_params_['min_samples_split'],
min_samples_leaf = clf_gbc.best_params_['min_samples_leaf'],
max_features = clf_gbc.best_params_['max_features'],
criterion = clf_gbc.best_params_['criterion'],
subsample =  clf_gbc.best_params_['subsample'])

y_pred_test = clf_gbc_1.predict(X_test) 
np.mean(y_pred_test  == y_test) # clf_rf_2.score(X_test,y_test)

y_pred_train = clf_gbc_1.predict(X_train) 
np.mean(y_pred_train == y_train) # clf_rf_2.score(X_train,y_train)

#RandomForestClassifier:
rf_par = {'n_estimators': list(range(200,1000,50)),
          'max_depth':list(range(1,5)),
          'n_jobs':[-1],
          'class_weight':[{0:1,1:2},{0:1,1:5}]
          }

clf_rf = GridSearchCV(RandomForestClassifier(), rf_par, cv=kfold)
clf_rf.fit(X_train,y_train)

clf_rf_1 = RandomForestClassifier(
n_estimators=clf_rf.best_params_['n_estimators'],
max_depth = clf_rf.best_params_['max_depth'],
n_jobs = clf_rf.best_params_['n_jobs'],
class_weight = clf_rf.best_params_['class_weight'],
random_state = 0)

clf_rf_1.fit(X_train,y_train)

y_pred_test = clf_rf_1.predict(X_test) 
np.mean(y_pred_test  == y_test) # clf_rf_2.score(X_test,y_test)

y_pred_train = clf_rf_1.predict(X_train) 
np.mean(y_pred_train == y_train) # clf_rf_2.score(X_train,y_train)

#################################################################

feature_importances = pd.DataFrame(
clf_rf.feature_importances_,
index = X_train.columns,
columns = ['importance']).sort_values('importance', ascending = False)

print(feature_importances.count())

feat_imp = feature_importances.sort_values(ascending = True, by='importance')
feat_imp.plot(kind='barh', title='Variaveis', grid = False)

df.corr().abs()['dano_na_plantacao'].sort_values(ascending = False)
