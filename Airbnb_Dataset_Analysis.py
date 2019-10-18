
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplt
from matplotlib import pyplot, pylab
import statistics
from matplotlib.pyplot import *
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
from pandas import Series
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn import linear_model as lm
import seaborn
from matplotlib import rcParams


# In[122]:


#read the csv file

listings = pd.read_csv("listings.csv")
listings.head()


# In[123]:


#replace the $ to calculate further.

listings['price'] = listings['price'].str.replace("$","")
listings['price'] = listings['price'].str.replace(",","")
listings['price'] = listings['price'].astype(float)
listings['price'].head()


# In[124]:


#Concatinate

data1 = pd.concat([listings['accommodates'],listings['price'],listings['availability_365'],
                   listings['number_of_reviews'],listings['review_scores_rating'],listings['bedrooms'],listings['bathrooms'],
                  listings['beds']],axis =1)
data1.head()


# In[125]:


# Correlation

list_cor = pd.DataFrame(data = data1)
list_cor.head()


# In[126]:


type(list_cor['price'])


# In[127]:


list_cor.corr(method='pearson')


# In[128]:


#boxplot for different neighborhood.

listings.boxplot(column='price',by= 'neighbourhood_group_cleansed', figsize=(30,20))


# In[129]:


#box plot using library seaborn

seaborn.boxplot(x="neighbourhood_group_cleansed", y="price", data=listings, linewidth=2.5)
rcParams['figure.figsize'] = 23.5,15
pyplt.show()


# In[130]:


# Multivariate plots to check relation between the variables

newlist = list_cor.iloc[:,[0,1,2,3,4,5,6,7]]
axes = pd.plotting.scatter_matrix(newlist, alpha=0.90)
pyplt.tight_layout()
pyplt.show()


# In[220]:


#scatter plot
y=listings['price']
x=listings['accommodates']
pyplt.figure(figsize=(10,10))

pyplt.scatter(x,y)
pyplt.xlabel("Accomodates No of People")
pyplt.ylabel("Price for Accomodation")


pyplt.show()


# In[219]:


#scatter plot
y1=listings['price']
x1=listings['beds']
pyplt.figure(figsize=(10,10))

pyplt.scatter(x1,y1)
pyplt.xlabel("No of Beds")
pyplt.ylabel("Price for Accomodation")

pyplt.show()


# In[218]:


# Fitting the model with Price vs Accommodates

mod_linear = ols('price ~ accommodates',listings).fit()

minaccommodates = min(listings['accommodates'])
maxaccommodates = max(listings['accommodates'])
xfit = pd.Series(np.arange(minaccommodates, maxaccommodates, 1),name = 'accommodates')
y_linearfit = mod_linear.predict(xfit)

pyplt.scatter(x,y, marker = "^", label = 'Raw values')
pyplt.plot(xfit,y_linearfit, color='r', label = 'Linear fit')


pyplt.xlabel('No of Accommodates')
pyplt.ylabel('Price')

pyplt.legend()
pyplt.show()

print(mod_linear.summary())


# In[134]:


# creating dataframes with the sum of each property type for Pie chart

proptype = listings['property_type']

total = proptype.size
total


# In[135]:


noofapart = np.sum(proptype.str.count('Apartment'))
print(noofapart)


# In[136]:


noofcabin = np.sum(proptype.str.count('Cabin'))


# In[137]:


noofhouse = np.sum(proptype.str.count('House'))


# In[138]:


noofLoft = np.sum(proptype.str.count('Loft'))


# In[139]:


noofBedBreakfast = np.sum(proptype.str.count('Bed & Breakfast'))


# In[140]:


noofOther = np.sum(proptype.str.count('Other')) + np.sum(proptype.str.count('Dorm'))+np.sum(proptype.str.count('Treehouse'))+np.sum(proptype.str.count('Tent'))+np.sum(proptype.str.count('Chalet'))
+np.sum(proptype.str.count('Camper/RV'))+np.sum(proptype.str.count('Boat'))+np.sum(proptype.str.count('Yurt'))


# In[141]:


noofBungalow = np.sum(proptype.str.count('Bungalow'))


# In[142]:


noofTownhouse = np.sum(proptype.str.count('Townhouse'))


# In[143]:


noofCondominium = np.sum(proptype.str.count('Condominium'))


# In[221]:


#other includes boats,camper/rv,dorm,treehouse,tent,chalet,yurt due to small sample size.

sizes = [noofhouse,noofcabin,noofBedBreakfast,noofBungalow,noofCondominium,noofLoft,noofOther,
         noofTownhouse,noofapart]

Label = ['House','Cabin','BedBreakfast','Bungalow','Condominium','Loft','Other','Townhouse','Apartment']
explode = (0.1,0,0,0,0,0,0,0,0.1)

pyplt.figure(figsize=(15,15))

pyplt.pie(sizes ,labels = Label, explode=explode, autopct='%1.1f%%')
pyplt.show()


# In[145]:


#Calculating mean for thhe price of different property type


prophouse = listings[listings['property_type'] == 'House']


# In[146]:


np.mean(prophouse['price'])


# In[147]:


propapart = listings[listings['property_type'] == 'Apartment']


# In[148]:


np.mean(propapart['price'])


# In[149]:


proptent = listings[listings['property_type'] == 'Tent']


# In[150]:


np.mean(proptent['price'])


# In[151]:


propbedbrk = listings[listings['property_type'] == 'Bed & Breakfast']


# In[152]:


np.mean(propbedbrk['price'])


# In[153]:


propboat = listings[listings['property_type'] == 'Boat']


# In[154]:


np.mean(propboat['price'])


# In[155]:


propbungalow = listings[listings['property_type'] == 'Bungalow']


# In[156]:


np.mean(propbungalow['price'])


# In[157]:


propcabin = listings[listings['property_type'] == 'Cabin']


# In[158]:


np.mean(propcabin['price'])


# In[159]:


propchalet = listings[listings['property_type'] == 'Chalet']


# In[160]:


np.mean(propchalet['price'])


# In[161]:


propCamperRV = listings[listings['property_type'] == 'Camper/RV']


# In[162]:


np.mean(propCamperRV['price'])


# In[163]:


propCondo = listings[listings['property_type'] == 'Condominium']


# In[164]:


np.mean(propCondo['price'])


# In[165]:


propth = listings[listings['property_type'] == 'Townhouse']


# In[166]:


np.mean(propth['price'])


# In[167]:


proploft = listings[listings['property_type'] == 'Loft']


# In[168]:


np.mean(proploft['price'])


# In[169]:


propdorm = listings[listings['property_type'] == 'Dorm']


# In[170]:


np.mean(propdorm['price'])


# In[171]:


proptreeh = listings[listings['property_type'] == 'Treehouse']


# In[172]:


np.mean(proptreeh['price'])


# In[173]:


propyurt = listings[listings['property_type'] == 'Yurt']


# In[174]:


np.mean(propyurt['price'])


# In[175]:


propoth = listings[listings['property_type'] == 'Other']


# In[176]:


np.mean(propoth['price'])


# In[177]:


# Bar chart for different property type.

typeaccomodation = ('Apartment', 'House', 'Cabin', 'Chalet', 'Bed & Breakfast', 'Other', 'Tent','Dorm','Yurt','Camper/RV','Treehouse',
           'Townhouse','Loft','Bungalow','Boat','Condominium')
series = np.arange(len(typeaccomodation))

meanprice = [np.mean(propapart['price']),np.mean(prophouse['price']),np.mean(propcabin['price']),np.mean(propchalet['price']),
            np.mean(propbedbrk['price']),np.mean(propoth['price']),np.mean(proptent['price']),np.mean(propdorm['price']),
             np.mean(propyurt['price']),np.mean(propCamperRV['price']),np.mean(proptreeh['price']),np.mean(propth['price']),
            np.mean(proploft['price']),np.mean(propbungalow['price']),np.mean(propboat['price']),np.mean(propCondo['price'])]

pyplt.figure(figsize=(25,15))

pyplt.bar(series, meanprice, align='center', alpha=0.5)
pyplt.xticks(series, typeaccomodation)

pyplt.xlabel('Type of accomodation')
pyplt.ylabel('Avg Price')
pyplt.show()


# In[178]:


queenan = listings[listings['neighbourhood_group_cleansed']=='Queen Anne']
np.mean(queenan['price'])


# In[179]:


ballard = listings[listings['neighbourhood_group_cleansed']=='Ballard']
np.mean(ballard['price'])


# In[180]:


others = listings[listings['neighbourhood_group_cleansed']=='Other neighborhoods']
np.mean(others['price'])


# In[181]:


cascade = listings[listings['neighbourhood_group_cleansed']=='Cascade']
np.mean(cascade['price'])


# In[182]:


central = listings[listings['neighbourhood_group_cleansed']=='Central Area']
np.mean(central['price'])


# In[183]:


university = listings[listings['neighbourhood_group_cleansed']=='University District']
np.mean(university['price'])


# In[184]:


downtown = listings[listings['neighbourhood_group_cleansed']=='Downtown']
np.mean(downtown['price'])


# In[185]:


magnolia = listings[listings['neighbourhood_group_cleansed']=='Magnolia']
np.mean(magnolia['price'])


# In[186]:


wseattle = listings[listings['neighbourhood_group_cleansed']=='West Seattle']
np.mean(wseattle['price'])


# In[187]:


Ibay = listings[listings['neighbourhood_group_cleansed']=='Interbay']
np.mean(Ibay['price'])


# In[188]:


bhill = listings[listings['neighbourhood_group_cleansed']=='Beacon Hill']
np.mean(bhill['price'])


# In[189]:


rainier = listings[listings['neighbourhood_group_cleansed']=='Rainier Valley']
np.mean(rainier['price'])


# In[190]:


delr = listings[listings['neighbourhood_group_cleansed']=='Delridge']
np.mean(delr['price'])


# In[191]:


seward = listings[listings['neighbourhood_group_cleansed']=='Seward Park']
np.mean(seward['price'])


# In[192]:


chill = listings[listings['neighbourhood_group_cleansed']=='Capitol Hill']
np.mean(chill['price'])


# In[193]:


ngate = listings[listings['neighbourhood_group_cleansed']=='Northgate']
np.mean(ngate['price'])


# In[194]:


lake = listings[listings['neighbourhood_group_cleansed']=='Lake City']
np.mean(lake['price'])


# In[195]:


# Bar chart for different neighbourhoods


area = ('Queen Anne', 'Ballard','Other neighborhoods','Cascade', 'Central Area', 'University District', 'Downtown', 'Magnolia','West Seattle','Interbay',
        'Beacon Hill','Rainer Valley','Delridge','Seward Park','Capitol Hill','Northgate','Lake City')

series = np.arange(len(area))

meanpricearea = [np.mean(queenan['price']),np.mean(ballard['price']),np.mean(others['price']),np.mean(cascade['price']),
                 np.mean(central['price']),np.mean(university['price']),np.mean(downtown['price']),np.mean(magnolia['price']),
                 np.mean(wseattle['price']),np.mean(Ibay['price']),np.mean(bhill['price']),np.mean(rainier['price']),
                 np.mean(delr['price']),np.mean(seward['price']),np.mean(chill['price']),np.mean(ngate['price']),np.mean(lake['price'])]
                 
pyplt.figure(figsize=(25,15))

pyplt.bar(series, meanpricearea, align='center', alpha=0.5)
pyplt.xticks(series, area)

pyplt.xlabel('Neighborhood')
pyplt.ylabel('Avg Price')
pyplt.show()


# In[196]:


#Boxplot for property type

seaborn.boxplot(x="property_type", y="price", data=listings, linewidth=3.5)
rcParams['figure.figsize'] = 27.5,15.5
pyplt.show()


# In[197]:


dfaov = pd.concat([np.log10(listings['price']),listings['neighbourhood_group_cleansed'],listings['property_type'],
                   listings['accommodates'],listings['room_type']], axis=1)
dfaov.head()


# In[198]:


# One way Anova

model = ols('price ~ C(property_type)+C(neighbourhood_group_cleansed)',dfaov).fit()
                
aov = sm.stats.anova_lm(model, typ=2)
print(aov)
model.params
print(model.summary())


# In[199]:


dfaovtownhouse = pd.concat([np.log10(propth['price']),propth['property_type']], axis=1)

dfaovcondo = pd.concat([np.log10(propCondo['price']),propCondo['property_type']], axis=1)

dfaovapart = pd.concat([np.log10(propapart['price']),propapart['property_type']], axis=1)

dfaovhouse = pd.concat([np.log10(prophouse['price']),prophouse['property_type']], axis=1)


dfaovtownhousecondo = pd.concat([dfaovtownhouse,dfaovcondo], axis = 0)

dfaovaparthouse = pd.concat([dfaovapart,dfaovhouse], axis = 0)


# In[200]:


# One way Anova between townhouse and condo

model1 = ols('price ~ C(property_type)',dfaovtownhousecondo).fit()
                
aov = sm.stats.anova_lm(model1, typ=2)
print(aov)
model1.params
print(model1.summary())


# In[201]:


# One way Anova between apartment and house

model2 = ols('price ~ C(property_type)',dfaovaparthouse).fit()
                
aov = sm.stats.anova_lm(model2, typ=2)
print(aov)
model2.params
print(model2.summary())


# In[202]:


#t test

sm.stats.ttest_ind(np.log10(prophouse['price']),np.log10(propapart['price']))


# In[203]:


#t test

sm.stats.ttest_ind(np.log10(propth['price']),np.log10(propCondo['price']))


# In[228]:


#histograms

nbins = 5
n, bins, patches = pyplt.hist(dfaovapart['price'], facecolor='green', alpha=0.5) 
title("Histogram of Price(in log scale) for Apartment")
legend()
pyplt.show()


# In[229]:


nbins = 5
n, bins, patches = pyplt.hist(dfaovhouse['price'], facecolor='red', alpha=0.5)
title("Histogram of Price(in log scale) for House")
legend()
pyplt.show()


# In[235]:


nbins = 5
n, bins, patches = pyplt.hist(np.log10(listings['price']), facecolor='orange', alpha=0.5)
title("Histogram of Price(in log scale) for all properties")
legend()
pyplt.show()


# In[207]:


# reading calendar data set for time series

calendar = pd.read_csv("calendar.csv",parse_dates=["date"],index_col="date")

calendar['price'] = calendar['price'].str.replace("$","")
calendar['price'] = calendar['price'].str.replace(",","")
calendar['price'] = calendar['price'].astype(float)

calendar['available']=calendar['available'].str.replace("t","1")
calendar['available']=calendar['available'].str.replace("f","0")
calendar['available'] = calendar['available'].astype(int)


calendar.head()


# In[208]:


#avg price for 2016
calendarnew = calendar["2016-01-01":"2016-12-31"]
calendarnew

calendarnew.price.resample('D').mean()


# In[209]:


#Plotting the avg price over time (Weekly)

get_ipython().run_line_magic('matplotlib', 'inline')
calendarnew.price.resample('W').mean().plot()


# In[217]:


#Plotting the avg price over time (Daily)

priceseries = calendarnew.price.resample('D').mean()

pyplt.figure(figsize=(20,10))

pyplt.plot(priceseries, label='Original Avg price')

# Moving avg filter - smoothing original avg price over time
smoothpriceseries = priceseries.rolling(10).mean()

pyplt.plot(smoothpriceseries, label='Moving avg smoothing (window size 10)')


#Exponential Smoothing
expsmoothpriceseries = priceseries.ewm(alpha=0.3).mean()

pyplt.plot(expsmoothpriceseries, label='Exponential Smoothing (alpha=0)')

pyplt.xlabel('Date')
pyplt.ylabel('Avg Price')


addlegend = pyplt.legend()


# In[211]:


calendarnew.available.resample('W').mean()


# In[212]:


get_ipython().run_line_magic('matplotlib', 'inline')
availabilitypercentage = 100*calendarnew.available.resample('W').mean()
availabilitypercentage.plot()


# In[213]:


#Plotting availability over time 

availabilitypercentage = 100*calendarnew.available.resample('D').mean()

pyplt.figure(figsize=(20,10))

pyplt.plot(availabilitypercentage, label='Original availability percentage')

# Moving avg filter - smoothing original availability percentage over time
smoothavailpercent = availabilitypercentage.rolling(10).mean()

pyplt.plot(smoothavailpercent, label='Moving avg smoothing (window size 10)')


#Exponential Smoothing
expsmoothavailpercent = availabilitypercentage.ewm(alpha=0.3).mean()

pyplt.plot(expsmoothavailpercent, label='Exponential Smoothing (alpha=0)')

pyplt.xlabel('Date')
pyplt.ylabel('Availability percentage')


addlegend = pyplt.legend()

