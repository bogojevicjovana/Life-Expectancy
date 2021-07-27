# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:51:18 2021

@author: Jovana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.stats.mstats import winsorize
import seaborn as sb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import tree

pd.set_option('display.float_format', lambda x: '%.2f' % x)

data_set = pd.read_csv("Life Expectancy Data.csv")

data_set.head()

data_set.info()

print(data_set.iloc[0])

a = data_set['Country'].unique()

data_set.shape
#2938 uzoraka, a 22 obelezja
#obelezja koja nemaju nedostajuce vrijednosti:
#Country, Year, Status, infant deaths, percentage expenditure, Measles, under-five deaths, HIV/AIDS

print("Atribut Country ima: ", data_set['Country'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Country'].isna().sum()/len(data_set)*100, '%')
print("Atribut Life expectancy ima: ", data_set['Life expectancy '].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Life expectancy '].isna().sum()/len(data_set)*100, '%')
print("Atribut Adult Mortality ima: ", data_set['Adult Mortality'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Adult Mortality'].isna().sum()/len(data_set)*100, '%')
print("Atribut infant deaths ima: ", data_set['infant deaths'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['infant deaths'].isna().sum()/len(data_set)*100, '%')
print("Atribut Alcohol ima: ", data_set['Alcohol'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Alcohol'].isna().sum()/len(data_set)*100, '%')
print("Atribut percentage expenditure ima: ", data_set['percentage expenditure'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['percentage expenditure'].isna().sum()/len(data_set)*100, '%')
print("Atribut Hepatitis B ima: ", data_set['Hepatitis B'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Hepatitis B'].isna().sum()/len(data_set)*100, '%')
print("Atribut BMI ima: ", data_set[' BMI '].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set[' BMI '].isna().sum()/len(data_set)*100, '%')
print("Atribut under-five deaths ima: ", data_set['under-five deaths '].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['under-five deaths '].isna().sum()/len(data_set)*100, '%')
print("Atribut Polio ima: ", data_set['Polio'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Polio'].isna().sum()/len(data_set)*100, '%')
print("Atribut 'Total expenditure' ima: ", data_set['Total expenditure'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Total expenditure'].isna().sum()/len(data_set)*100, '%')
print("Atribut Diphtheria ima: ", data_set['Diphtheria '].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Diphtheria '].isna().sum()/len(data_set)*100, '%')
print("Atribut HIV/AIDS ima: ", data_set[' HIV/AIDS'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set[' HIV/AIDS'].isna().sum()/len(data_set)*100, '%')
print("Atribut GDP ima: ", data_set['GDP'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['GDP'].isna().sum()/len(data_set)*100, '%')
print("Atribut Population ima: ", data_set['Population'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Population'].isna().sum()/len(data_set)*100, '%')
print("Atribut thinness  1-19 years ima: ", data_set[' thinness  1-19 years'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set[' thinness  1-19 years'].isna().sum()/len(data_set)*100, '%')
print("Atribut thinness 5-9 years ima: ", data_set[' thinness 5-9 years'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set[' thinness 5-9 years'].isna().sum()/len(data_set)*100, '%')
print("Atribut Income composition of resources ima: ", data_set['Income composition of resources'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Income composition of resources'].isna().sum()/len(data_set)*100, '%')
print("Atribut Schooling  ima: ", data_set['Schooling'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['Schooling'].isna().sum()/len(data_set)*100, '%')

print(data_set['Year'].unique())
   
#rjesavanje problema sa nan vrijednostima

alcohol = data_set[['Alcohol', 'Year']]

# skoro za sve drzave nedostaju vrijednosti za obelezje 'Alcohol' za 2015. godinu
# moguce rjesenje je da uzmem vrijednost od prosle godine, tj. sa 2014. godinu
alcohol_nan = alcohol[alcohol['Alcohol'].isna()]
data_set['Alcohol'].fillna(method = 'bfill', inplace = True) #za 2015. godinu dodate su vrijednosti od 2014. godine

data_set['Alcohol'].fillna(method = 'bfill', inplace = True)
data_set['Alcohol'].fillna(method = 'bfill', inplace = True)

data_set_sudan = data_set[data_set['Country'] == "South Sudan"]
data_set_sudan.shape
#izbacena je drzava South Sudan:
#ima "validnih" kolona, ostalo su nan kolone i kolone u kojima se nalaze 0 
data_set = data_set.loc[(data_set["Country"] != 'South Sudan')]


#takodje za vecinu drzava nedostaju vrijednosti za 2015. godinu, za drzavu Iraq nedostaju vrijednosti za 2000, 2001 i 2002
total_expenditure = data_set[['Country', 'Year', 'Total expenditure']]
total_expenditure_nan = total_expenditure[total_expenditure['Total expenditure'].isna()]
data_set_somalia = data_set[data_set['Country'] == "Somalia"]

hepatitisB = data_set[['Country', 'Year', 'Hepatitis B']].sort_values(by = (['Country', 'Year']), ascending=True)
hepatitisB_nan = hepatitisB[hepatitisB['Hepatitis B'].isna()]

population = data_set[['Country', 'Year', 'Population']].sort_values(by = (['Country', 'Year']), ascending=True)

life_expentancy = data_set[['Country', 'Year', 'Life expectancy ']]
life_expentancy_nan = life_expentancy[life_expentancy['Life expectancy '].isna()]

data_set_d1 = data_set[data_set['Country'] == "Tuvalu"]

#izbacene su ove drzave jer imaju samo jednu vrstu
data_set = data_set.loc[(data_set["Country"] != 'Cook Islands')]
data_set = data_set.loc[(data_set["Country"] != 'San Marino')]
data_set = data_set.loc[(data_set["Country"] != 'Dominica')]
data_set = data_set.loc[(data_set["Country"] != 'Monaco')]
data_set = data_set.loc[(data_set["Country"] != 'Nauru')]
data_set = data_set.loc[(data_set["Country"] != 'Niue')]
data_set = data_set.loc[(data_set["Country"] != 'Palau')]
data_set = data_set.loc[(data_set["Country"] != 'Saint Kitts and Nevis')]
data_set = data_set.loc[(data_set["Country"] != 'Tuvalu')]
data_set = data_set.loc[(data_set["Country"] != 'Marshall Islands')]

#ove zemlje nemaju informacije od 2000. do 2015. o Hepatitis B obelezju
data_set = data_set.loc[(data_set["Country"] != 'Finland')]
data_set = data_set.loc[(data_set["Country"] != 'Hungary')]
data_set = data_set.loc[(data_set["Country"] != 'Iceland')]
data_set = data_set.loc[(data_set["Country"] != 'Japan')]
data_set = data_set.loc[(data_set["Country"] != 'Norway')]
data_set = data_set.loc[(data_set["Country"] != 'Slovenia')]
data_set = data_set.loc[(data_set["Country"] != 'Switzerland')]
data_set = data_set.loc[(data_set["Country"] != 'United Kingdom of Great Britain and Northern Ireland')]

data_set = data_set.loc[(data_set["Country"] != "Côte d'Ivoire")]
data_set = data_set.loc[(data_set["Country"] != 'Czechia')]
data_set = data_set.loc[(data_set["Country"] != "Democratic People's Republic of Korea")]
data_set = data_set.loc[(data_set["Country"] != 'Democratic Republic of the Congo')]
data_set = data_set.loc[(data_set["Country"] != 'Congo')]
data_set = data_set.loc[(data_set["Country"] != 'Egypt')]
data_set = data_set.loc[(data_set["Country"] != 'Plurinational State of')]
data_set = data_set.loc[(data_set["Country"] != 'Gambia')]
data_set = data_set.loc[(data_set["Country"] != "Iran (Islamic Republic of)")]
data_set = data_set.loc[(data_set["Country"] != 'Gambia')]
data_set = data_set.loc[(data_set["Country"] != 'Kyrgyzstan')]
data_set = data_set.loc[(data_set["Country"] != 'Country Micronesia (Federated States of)')]
data_set = data_set.loc[(data_set["Country"] != "Lao People's Democratic Republic")]
data_set = data_set.loc[(data_set["Country"] != 'Republic of Moldova')]
data_set = data_set.loc[(data_set["Country"] != "Saint Lucia")]
data_set = data_set.loc[(data_set["Country"] != 'Republic of Korea')]
data_set = data_set.loc[(data_set["Country"] != 'Saint Vincent and the Grenadines')]
data_set = data_set.loc[(data_set["Country"] != 'Slovakia')]
data_set = data_set.loc[(data_set["Country"] != 'Sudan')]	
data_set = data_set.loc[(data_set["Country"] != 'The former Yugoslav republic of Macedonia')]		
data_set = data_set.loc[(data_set["Country"] != 'United Republic of Tanzania')]	
data_set = data_set.loc[(data_set["Country"] != 'United States of America')]	
data_set = data_set.loc[(data_set["Country"] != "Bolivia (Plurinational State of)")]
data_set = data_set.loc[(data_set["Country"] != "Micronesia (Federated States of)")]
data_set = data_set.loc[(data_set["Country"] != "Bolivia (Plurinational State of)")]
data_set = data_set.loc[(data_set["Country"] != 'Somalia')]
data_set = data_set.loc[(data_set["Country"] != 'Venezuela (Bolivarian Republic of)')]
data_set = data_set.loc[(data_set["Country"] != 'Viet Nam')]
data_set = data_set.loc[(data_set["Country"] != 'Montenegro')]
data_set = data_set.loc[(data_set["Country"] != 'Yemen')]

data_set['Hepatitis B'].fillna(method = 'ffill', inplace = True)
data_set['GDP'].fillna(method = 'ffill', inplace = True)
data_set['Population'].fillna(method = 'ffill', inplace = True)
data_set['Polio'].fillna(method = 'ffill', inplace = True)
data_set['Diphtheria '].fillna(method = 'ffill', inplace = True)
data_set['Total expenditure'].fillna(method = 'bfill', inplace = True)

data_set.loc[data_set['Country'] == 'Antigua and Barbuda', ['Population']] = 85000
data_set.loc[data_set['Country'] == 'Bahamas', ['GDP']] = 31158
data_set.loc[data_set['Country'] == 'Bahamas', ['Population']] = 361135
data_set.loc[data_set['Country'] == 'Bahrain', ['Population']] = 1115399
data_set.loc[data_set['Country'] == 'Barbados', ['Population']] = 265620
data_set.loc[data_set['Country'] == 'Brunei Darussalam', ['Population']] = 375458
data_set.loc[data_set['Country'] == 'Cuba', ['Population']] = 11234657
data_set.loc[data_set['Country'] == 'Grenada', ['Population']] = 105500
data_set.loc[data_set['Country'] == 'Kuwait', ['Population']] = 2785699
data_set.loc[data_set['Country'] == 'Libya', ['Population']] = 5943121
data_set.loc[data_set['Country'] == 'New Zealand', ['Population']] = 5943121
data_set.loc[data_set['Country'] == 'Oman', ['Population']] = 3022001
data_set.loc[data_set['Country'] == 'Qatar', ['Population']] = 1469980
data_set.loc[data_set['Country'] == 'Saudi Arabia', ['Population']] = 5943121
data_set.loc[data_set['Country'] == 'Singapore', ['Population']] = 4754470
data_set.loc[data_set['Country'] == 'United Arab Emirates', ['Population']] = 6383793

data_set.loc[data_set['Country'] == 'Sao Tome and Principe', ['GDP']] = data_set.loc[data_set['Country'] == 'Sao Tome and Principe', ['GDP']].fillna(51.216381)

data_set['Status'].unique()
data_set.loc[data_set['Status']=='Developing','Status']= 0
data_set.loc[data_set['Status']=='Developed','Status']= 1


# =============================================================================
#analiza obelezja: statisticki parametri i korelacija izmedju obelezja

columns = {'Life expectancy ': 1 , 'Adult Mortality': 2 , 'Alcohol': 3, 'percentage expenditure': 4, 'Hepatitis B': 5,
       'Measles ' : 6, ' BMI ': 7, 'under-five deaths ' : 8, 'Polio' : 9, 'Total expenditure' :10, 'Diphtheria ':11, 
       ' HIV/AIDS':12, 'GDP':13, 'Population' : 14, ' thinness  1-19 years' : 15, ' thinness 5-9 years' : 16,
       'Income composition of resources' : 17, 'Schooling' : 18, 'infant deaths': 19}

statisticki_parametri = data_set.describe()
print(statisticki_parametri)

ddt = data_set.groupby(['Country']).mean().sort_values(by = 'Life expectancy ', ascending=True)

plt.figure(figsize=(6,6))
plt.bar(data_set.groupby('Status')['Status'].count().index,data_set.groupby('Status')['Life expectancy '].mean())
plt.xlabel("Status",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Status")
plt.show()

plt.figure(figsize = (50, 7))
le_country = data_set.groupby('Country')['Life expectancy '].mean().sort_values(ascending=True)
le_country.plot(kind='bar', fontsize=25)
plt.title("Prosečna životna starost po državama",fontsize=40)
plt.xlabel("Country",fontsize=35)
plt.ylabel("Avg Life_Expectancy",fontsize=35)
plt.show()

plt.figure(figsize=(35,40))
plt.grid()
for variable, i in columns.items():
    plt.subplot(5 ,4 ,i)
    plt.boxplot(data_set[variable], whis = 1.5)
    plt.title(variable)
plt.show()

data_set.loc[data_set['infant deaths'] > 601, 'infant deaths'] = 600

plt.boxplot([data_set["infant deaths"]])
plt.yticks(np.arange(0, 1005, 200))
plt.xticks([1], ["infant deaths"])
plt.grid()


plt.boxplot([data_set["percentage expenditure"]])
plt.yticks(np.arange(0, 1020, 100))
plt.xticks([1], ["percentage expenditure"])
plt.grid()

data_set.loc[data_set['percentage expenditure'] > 1020, 'percentage expenditure'] = 1000

plt.figure
plt.boxplot([data_set["Measles "]])
plt.yticks(np.arange(0, 1000, 100))
plt.xticks([1], ["Measles"])
plt.grid()

data_set.loc[data_set['Measles '] > 1020, 'Measles '] = 1000

plt.figure
plt.boxplot([data_set['under-five deaths ']])
plt.yticks(np.arange(0, 2600, 400))
plt.xticks([1], ['Under five deaths '])
plt.grid()

data_set.loc[data_set['under-five deaths '] > 501, 'under-five deaths '] = 500


plt.boxplot([data_set["Life expectancy "], data_set[" BMI "], data_set["Hepatitis B"], data_set[" HIV/AIDS"], data_set["Polio"]])
plt.ylabel("Vrednosti obelezja")
plt.yticks(np.arange(0, 150, 10))
plt.xticks([1, 2, 3, 4, 5], ["Life expectancy ", "BMI", "Hepatitis B", "HIV/AIDS", "Polio"])
plt.grid()


#Q1 = data_set.quantile(0.25)
#print(Q1)
#Q3 = data_set.quantile(0.75)
#print(Q3)
#IQR = Q3 - Q1
#print(IQR)
#data_set = data_set[~((data_set < (Q1 - 1.5 * IQR)) | (data_set > (Q3 + 1.5 * IQR))).any(axis=1)]


#korelacija
corr = data_set.corr()
f = plt.figure(figsize=(20, 15))
sb.heatmap(corr, annot=True);

a = data_set[['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
       'infant deaths']]

sb.pairplot(a)

b = data_set[['Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Life expectancy ']]

sb.pairplot(b)

c = data_set[[ 'Polio', 'Total expenditure','Diphtheria ', ' HIV/AIDS', 'GDP', 'Life expectancy ']]

sb.pairplot(c)

plt.scatter(data_set["Life expectancy "], data_set["Adult Mortality"])
plt.xticks(np.arange(30, 100, 10))
plt.yticks(np.arange(0, 1000, 100))
plt.xlabel('Life expectancy ')
plt.ylabel('Adult Mortality')


plt.scatter(data_set["Life expectancy "], data_set["Alcohol"])
plt.xticks(np.arange(25, 100, 10))
plt.yticks(np.arange(0, 100, 10))
plt.xlabel('Life expectancy ')
plt.ylabel('Alcohol')

# =============================================================================
#model

#promjenljiva cije vrijednosti se predivdjaju
y = data_set['Life expectancy ']

#promjenljive na osnovu kojih ce se predvidjati vrijednosti
x = data_set.drop('Life expectancy ', axis = 1)

country_dummy = pd.get_dummies(x['Country'])

x.drop(['Country','Status'], inplace=True, axis=1)
x = pd.concat([x ,country_dummy], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)

def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(10))


#lasso regresija
parameters = {'alpha':[0.5, 0.7, 0.9, 1]}
rf = Lasso()
clf = GridSearchCV(rf, parameters)
clf.fit(x_train_std, y_train)
print(clf.best_score_)
print(clf.best_params_)

lasso_model = Lasso(alpha = 0.5)
lasso_model.fit(x_train_std, y_train)
y_predicted = lasso_model.predict(x_test_std)
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

#ridge regresija
parameters = {'alpha':[0.5, 0.7, 0.9, 1]}
rf = Ridge()
clf = GridSearchCV(rf, parameters)
clf.fit(x_train_std, y_train)
print(clf.best_score_)
print(clf.best_params_)

ridge_model = Ridge(alpha = 0.5)
ridge_model.fit(x_train_std, y_train)
y_predicted = ridge_model.predict(x_test_std)
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

#knn regresor
parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15], 'weights': ('uniform', 'distance'), 'metric':('hamming', 'jaccard', 'minkowski')}
rf = KNeighborsRegressor()
clf = GridSearchCV(rf, parameters)
clf.fit(x_train_std, y_train)
print(clf.best_score_)
print(clf.best_params_)

knn_model = KNeighborsRegressor(n_neighbors = 4, metric='minkowski', weights = 'distance')
knn_model.fit(x_train_std, y_train)
y_predicted = knn_model.predict(x_test_std)
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

#stabla odluke
parameters = {'criterion' : ('mse', 'mae'), 'max_depth' :[5, 10, 15], 
              'min_samples_split' : [0.01, 0.05]}
dt = DecisionTreeRegressor()
clf=GridSearchCV(dt, parameters)
clf.fit(x_train_std, y_train)
print(clf.best_score_)
print(clf.best_params_)

dt_model = DecisionTreeRegressor(max_depth = 15, criterion = 'mse', min_samples_split = 0.01)
dt_model.fit(x_train, y_train)
y_predicted = dt_model.predict(x_test)
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

plt.figure(figsize = (100, 100))
tree.plot_tree(dt_model)
plt.show()

#redukcija
pca = PCA(n_components=0.9)
pca.fit(x_train_std)
x_train_r = pca.transform(x_train_std)
x_test_r = pca.transform(x_test_std)
print('Redukovani prostor ima dimenziju: ', pca.n_components_)
   
#lasso nakon redukcije 
parameters = {'alpha':[0.5, 0.7, 0.9, 1]}
rf = Lasso()
clf = GridSearchCV(rf, parameters)
clf.fit(x_train_r, y_train)
print(clf.best_score_)
print(clf.best_params_)

lasso_model = Lasso(alpha = 0.5)
lasso_model.fit(x_train_r, y_train)
y_predicted = lasso_model.predict(x_test_r)
model_evaluation(y_test, y_predicted, x_train_r.shape[0], x_train_r.shape[1])

#ridge regresija
parameters = {'alpha':[0.5, 0.7, 0.9, 1]}
rf = Ridge()
clf = GridSearchCV(rf, parameters)
clf.fit(x_train_r, y_train)
print(clf.best_score_)
print(clf.best_params_)

ridge_model = Ridge(alpha = 1)
ridge_model.fit(x_train_r, y_train)
y_predicted = ridge_model.predict(x_test_r)
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

#knn regresor
parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15], 'weights': ('uniform', 'distance'), 'metric':('hamming', 'jaccard', 'minkowski')}
rf = KNeighborsRegressor()
clf = GridSearchCV(rf, parameters)
clf.fit(x_train_r, y_train)
print(clf.best_score_)
print(clf.best_params_)

knn_model = KNeighborsRegressor(n_neighbors = 3, metric='minkowski', weights = 'distance')
knn_model.fit(x_train_r, y_train)
y_predicted = knn_model.predict(x_test_r)
model_evaluation(y_test, y_predicted, x_train_r.shape[0], x_train_r.shape[1])

#stabla odluke
parameters = {'criterion' : ('mse', 'mae'), 'max_depth' :[5, 10, 15, 20], 
              'min_samples_split' : [0.01, 0.05]}
dt = DecisionTreeRegressor()
clf=GridSearchCV(dt, parameters)
clf.fit(x_train_r, y_train)
print(clf.best_score_)
print(clf.best_params_)

dt_model = DecisionTreeRegressor(max_depth = 10, criterion = 'mse', min_samples_split = 0.01)
dt_model.fit(x_train, y_train)
y_predicted = dt_model.predict(x_test)
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])









