#!/usr/bin/env python
# coding: utf-8

# # CSE:382 Data Mining and Business Intelligence

# Prepared by:
# Sohayla Ihab Hamed [19P7343]
# Youssef Mahmoud Massoud [18P8814]
# Salma Ihab Abdelmawgoud [19P8794]
# Youssef Hany Onsy [18P1789]

# Loan prediction is a common data mining problem which most retail banks solve on a 
# daily basis. It’s the process of predicting who deserves to receive a given loan and at what 
# rate based on certain characteristics of the borrower, be it an individual or a company. 
# Those characteristics are mined for their use in a risk assessment process to determine 
# the amount of risks the lender (the bank) will be incurring when loaning the particular 
# individual/company. 
# 
# For example, some banks can model their interest rate for lending 
# based on how much risk their model assumes a certain individual pose, and thereby 
# require higher interest rates for those who pose higher risks of default and vice versa.
# In our project, we use data mining techniques to analyze and predict whether a certain 
# individual can be allowed to take a loan from our bank or whether said individual shall be 
# denied the loan. This will be based on a set of features like marital status, education, 
# employments and other features within the restrictions of the dataset we were provided. 
# We will apply various data mining techniques to achieve our required goal where we end 
# up with a classification model that we will train and later test.
# 
# By the end of this project, we should have a set of divided preprocessing techniques(explained here)
# into functions and procedures, such that any real data may be handled

# # LOAN PREDICTION

# ## DATA VISUALIZATION

# In[1]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


TrainData = pd.read_csv("Train Data.csv", delimiter = ',', header = 0, index_col = 0)
#print(TrainData)


# In[3]:


# showing the univariant statistics for numerical variables
TrainData.describe()


# In[4]:


TrainData.info()


# In[5]:


TrainData.mode()


# In[6]:


# showing the bivariant statistics between each variable with the Loan Status(Yes or No)
TrainData.groupby("Loan_Status").describe()


# In[7]:


print("kurtosis: ")
TrainData.kurtosis(numeric_only=True)


# In[8]:


print("skew: ")
TrainData.skew(numeric_only=True)


# In[9]:


#Notes!! Conclusions to KEEP IN MIND: 
# CoapplicantIncome: 25% is $0. Later, remove tuples where CoapplicantIncome=0 and analyze on its own.
# Loan_Amount_term: Q1, Q2, Q3 = 360. Which means the data is skewed towards their favor. (360 months loan)
# STD is pretty high for ApplicantIncome=6109 and CoapplicantIncome=2926
# Credit_History: Q1, Q2, Q3 = 1. Which means that most people have credit history.
# Note: very important to treat loan term and credit history as categorical, not numerical, since its values are discrete

# skewness determines asymmetrical distribution
# · -0.5 < skewness < 0.5, the data are fairly symmetrical
# ·  -1 < skewness < — 0.5 or  0.5 < skewness < 1, the data are moderately skewed
# · skewness < -1 or skewness > 1, the data are highly skewed

# kurtosis determine the volume of the outlier
# · If the distribution is tall and thin it is called a leptokurtic distribution(Kurtosis > 3). Values in a leptokurtic distribution are near the mean or at the extremes.
# · A flat distribution where the values are moderately spread out (i.e., unlike leptokurtic) is called platykurtic(Kurtosis <3) distribution.
# · A distribution whose shape is in between a leptokurtic distribution and a platykurtic distribution is called a mesokurtic(Kurtosis=3) distribution. A mesokurtic distribution looks more close to a normal distribution.
# source: https://medium.com/@atanudan/kurtosis-skew-function-in-pandas-aa63d72e20de


# In[10]:


#Visualizing Numerical Features
NumericData = TrainData.select_dtypes(exclude = ['object']).columns.tolist()
TrainData_n = TrainData[NumericData]
#print("Description of Numerical Features:", TrainData.describe()) 


# seaborn histograms (univariate and pairwise)
# TODO tweak this so it looks better. try other things like scatter, density, histo, etc
# TODO please look up how to visualize correlation with the TARGET
# "Numerical features can be visualized by plotting their distribution and having a look at their statistical properties, 
# such as skewness and kurtosis. For categorical features, the distribution is better visualized using histograms. 
# Finally, you can calculate and visualize the correlation between the features and also the features with the target value."
# source: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
# tutorial (ref doc): https://seaborn.pydata.org/tutorial/distributions.html
# TODO please examine the correlation matrix results at the end of the notebook, 
# and draw a relational plot with the three most correlated values
# tutorial: https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial
# Important note: this is a very general API. You can adjust more specifically by using jointplot, scatterplot etc.
sns.pairplot(TrainData)

plt.show()


# In[11]:


#Visualizing Categorical Features
#TODO use countplot to visualize univariate categorical
sns.countplot(data=TrainData,
    x="Credit_History",
    hue="Loan_Status")
plt.show()


# In[12]:


sns.countplot(data=TrainData,
    x="Loan_Amount_Term",
    hue="Loan_Status")
plt.show()


# In[13]:


sns.displot(data=TrainData,
    x="ApplicantIncome", #y='ApplicantIncome',
    hue="Loan_Status")
plt.show()


# In[14]:


sns.displot(
    data=TrainData, #kind ="count",
    x="Married" , hue="Loan_Status"#, element="step"
)
plt.show()


# In[15]:


sns.catplot(
    data=TrainData, kind ="count",
    x="Gender" 
)
plt.show()


# In[16]:


sns.catplot(
    data=TrainData, kind ="count",
    y="Dependents", #hue="Loan_Status"#, palette="ch:.25" 
)
plt.show()


# In[17]:


sns.catplot(
    data=TrainData, kind ="count",
    x="Self_Employed"
)
plt.show()


# In[18]:


sns.catplot(
    data=TrainData, kind ="count",
    y="Property_Area", hue="Loan_Status"#, palette="ch:.25" 
)
plt.show()


# In[19]:


sns.catplot(
    data=TrainData, kind ="box", y="ApplicantIncome",
    x="Self_Employed", hue="Loan_Status", col="Credit_History"#, palette="ch:.25" 
)
plt.show()


# In[20]:


#catplot is used
sns.catplot(
    data=TrainData, kind ="box",
    x="LoanAmount", y="Dependents", 
    hue="Loan_Status" 
)
plt.show()
# When categories are harder to define, we will use binning methods.
# Scroll below to binning.


# ## DATA CLEANING

# ### Handle Missing Data

# Identify missing data of both types, numerical and categorical data

# In[21]:


NumericData = TrainData.select_dtypes(exclude = ['object']).columns.tolist()
TrainData_n = TrainData[NumericData]
#print(TrainData_n, '\n')

CategoricData = TrainData.select_dtypes(include = ['object']).columns.tolist()
TrainData_c = TrainData[CategoricData]
#print(TrainData_c, '\n')


# In[22]:


# Drop duplicates
print(TrainData.shape)
TrainData.drop_duplicates(inplace=True)
print(TrainData.shape)


# In[23]:


# Now, we try to figure out what kind of missing values are there. 
# “0”, “Not Applicable”, “NA”, “None”, “Null”, or “INF” all can mean that the value is missing.
for feature in TrainData.columns:
    res = list(set(TrainData[feature]))
    print(res)

# Therefore, all missing values are np.nan or None


# In[24]:


# First Approach: removing tuples. We remove tuples only if the dataset is large enough, and the tuple has multiple missing values.
# Complete Case Analysis(CCA) may be used if data is MAR (Missing At Random)

print(TrainData.isnull().sum()) #returns np.na or None
print(TrainData.shape)
TrainData_nonull = TrainData.dropna(inplace=False)
print(TrainData_nonull.shape)

# Credit_History is the feature with the most missing tuples, but 50/641 is not significant enough to discard the feature entirely.
# 480/641 tuples remain. That's only around 78.18% of data. 
# Around 21.8% of data tuples have missing values. This may not be the best approach. Once the classification models are completed, 
# we can test using both datasets and evaluate their accuracy.


# In[25]:


# Second Approach: Using conclusions from data visualization section, decide for each feature how to fill the missing values.
# Gender: most_frequent (male), since there is a huge variance between the two.
# Married: we can drop the 3 tuples, knowing, their effect over 614 tuples is not significant anyway.
# Dependents: knn
# Self-Employed: knn
# LoanAmount: (normally distributed variable, since mean=342, median=360, mode=360) impute with mean
# Loan_Amount_Term: impute with median
# Credit_History: knn



#TODO later, we can fill the missing values using a regression model. (if we have time, research)



# Note: In regards to imputing missing data: "Mean is most useful when the original data is not skewed, 
# while the median is more robust, not sensitive to outliers, and thus used when data is skewed.
# "It is worth mentioning that linear regression models are sensitive to outliers.
# "𝑘 nearest neighbour imputation, which classifies similar records and put them together, can also be utilized. 
# A missing value is then filled out by finding first the 𝑘 records closest to the record with missing values. 
# Next, a value is chosen from (or computed out of) the 𝑘 nearest neighbours."
# source: https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4
# "One thing to note here is that the KNN Imputer does not recognize text data values." We must discretize categorical features first.
# "Another critical point here is that the KNN Imptuer is a distance-based imputation method and it requires us to normalize our data. 
# Otherwise, the different scales of our data will lead the KNN Imputer to generate biased replacements for the missing values."
# source: https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e
# very IMPORTANT read, for later phases, on how to structure our preprocessing + predictors: (STACKING)
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py


# In[26]:


MostFreqImputer = SimpleImputer(missing_values = np.NaN , strategy = 'most_frequent') #cat and num
MeanImputer = SimpleImputer(missing_values = np.NaN , strategy = 'mean') #num
MedianImputer = SimpleImputer(missing_values = np.NaN , strategy = 'median') #num
#MedianCatImputer = SimpleImputer(missing_values = np.NaN , strategy = 'constant', fill_value="calc median for feature x") #cat


# In[27]:


# For Gender: mode
TrainData_Gender = np.array(TrainData['Gender']).reshape(-1, 1)
MostFreqImputer.fit(TrainData_Gender)
TrainData_Gender = MostFreqImputer.transform(TrainData_Gender)
TrainData_Gender = TrainData_Gender.flatten()
TrainData['Gender'] = TrainData_Gender


# In[28]:


# For LoanAmount: mean
TrainData_LoanA = np.array(TrainData['LoanAmount']).reshape(-1, 1)
MeanImputer.fit(TrainData_LoanA)
TrainData_LoanA = MeanImputer.transform(TrainData_LoanA)
TrainData_LoanA = TrainData_LoanA.flatten()
#print(TrainData_LoanA.shape)

TrainData['LoanAmount'] = TrainData_LoanA


# In[29]:


# For Loan_Amount_Term: median
TrainData_LoanT = np.array(TrainData['Loan_Amount_Term']).reshape(-1, 1)
MedianImputer.fit(TrainData_LoanT)
TrainData_LoanT = MedianImputer.transform(TrainData_LoanT)
TrainData_LoanT = TrainData_LoanT.flatten()
print(TrainData_LoanT.shape)
TrainData['Loan_Amount_Term'] = TrainData_LoanT


# In[30]:


# For Married: drop

import math

TrainData_Married = TrainData.copy()
print(TrainData_Married.isnull().sum())
print(TrainData_Married.shape)

TrainData_Married = TrainData_Married.drop(index=[row for row in TrainData_Married.index 
                                                  if pd.isna(TrainData_Married.loc[row, 'Married'])])

print(TrainData_Married.isnull().sum())
print(TrainData_Married.shape)

TrainData = TrainData_Married.copy()


# KNNImputer to impute Dependents, Self_Employed, and Credit_History

# Step 1) LabelEncoder

# In[31]:


# Dependents: knn
# Self-Employed: knn
# Credit_History: knn


# KNN (after discretization and normalization as discussed in the comments)

# To find KNN between gender and loan_status, first use encoding to convert all categoric data into numeric data
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
genderlabels = label_encoder.fit_transform(TrainData['Gender'])
dependentslabels = label_encoder.fit_transform(TrainData['Dependents'])
selfemployedlabels = label_encoder.fit_transform(TrainData['Self_Employed'])
credithistorylabels = label_encoder.fit_transform(TrainData['Credit_History'])
marriedlabels = label_encoder.fit_transform(TrainData['Married'])
educationlabels = label_encoder.fit_transform(TrainData['Education'])
propertylabels = label_encoder.fit_transform(TrainData['Property_Area'])
loanstatuslabels = label_encoder.fit_transform(TrainData['Loan_Status'])


unique = np.unique(genderlabels)
print("Gender labels: ", unique)
print(list(set(TrainData['Gender'])))

unique = np.unique(dependentslabels)
print("Dependents labels: ", unique)
print(list(set(TrainData['Dependents'])))

unique = np.unique(selfemployedlabels)
print("Self_Employed labels: ", unique)
print(list(set(TrainData['Self_Employed'])))

unique = np.unique(credithistorylabels)
print("Credit_History labels: ", unique)
print(list(set(TrainData['Credit_History'])))

unique = np.unique(marriedlabels)
print("Married labels: ", unique)
print(list(set(TrainData['Married'])))

unique = np.unique(educationlabels)
print("Education labels: ", unique)
print(list(set(TrainData['Education'])))

unique = np.unique(propertylabels)
print("Property_Area labels: ", unique)
print(list(set(TrainData['Property_Area'])))

unique = np.unique(loanstatuslabels)
print("Loan_Status labels: ", unique)
print(list(set(TrainData['Loan_Status'])))


#Add to numeric table
TrainData_encoded = TrainData.copy()
TrainData_encoded['Gender'] = genderlabels  
TrainData_encoded['Dependents'] = dependentslabels  
TrainData_encoded['Self_Employed'] = selfemployedlabels  
TrainData_encoded['Credit_History'] = credithistorylabels  
TrainData_encoded['Married'] = marriedlabels  
TrainData_encoded['Education'] = educationlabels  
TrainData_encoded['Property_Area'] = propertylabels  
TrainData_encoded['Loan_Status'] = loanstatuslabels  



TrainData_encoded


# Step 2) MinMaxScaler

# In[32]:


# Before implementing KNN, we also need to normalize our data. For simplicity, we will use MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
TrainData_knn = pd.DataFrame(scaler.fit_transform(TrainData_encoded), columns = TrainData_encoded.columns)
ax = sns.displot(TrainData_knn, kind = "kde",color = "#e64e4e", height=10, aspect=2,
            linewidth = 3 )
ax.fig.suptitle('Density after minmax scaling between [0,1]', size = 20)
plt.show(ax)

TrainData_knn


# Step 3) KNNImputer

# In[33]:


# Finally, KNNImputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
TrainData_knn = pd.DataFrame(imputer.fit_transform(TrainData_knn),columns = TrainData_knn.columns)

TrainData = TrainData_knn.copy()
TrainData.isna().sum() # 0 for all! Missing values are no more. Data cleaned.


# In[34]:


# Third Approach: Flagging
# Some data is missing not at random (MNAR). This means that the data is probably missing due to the feature itself.
# One example would be loan amount. If it is too high, people might refuse to say.
# Some data is missing at random. This means that the data is probably missing because of another measured variable.
# An example would be self-employment. Those working in rural areas such as farms might be trivially self-employed. We might
# trivially replace all the missing values with "No".
# Observing the data, credit history is the variable with most missing values. This may be due to applicants not being able
# to prove their credit history.
# In this case, data is missing completely at random (MCAT).
# We mention this approach, even though it may not be applicable to our project, because it is important to remember
# that even missing values can provide valuable information, which we may get rid of by imputing.


# In[35]:


# Filled in missing values (using second approach)

print("example record: ", TrainData.values[0]) #example of record with missing num values
print("example record: ", TrainData.values[23]) #example of record with missing cat values
TrainData


# Step 4) Normalization Transform within range [0, 1]

# In[36]:


from sklearn.preprocessing import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

TrainDatacopy = TrainData.copy()
TrainDatacopy = pd.DataFrame(Normalizer().fit_transform(TrainData), columns = TrainData_knn.columns)
#print(TrainDatatransform)
#print(TrainDatacopy)

ax2 = sns.displot(TrainDatacopy, kind = "kde",color = "#e64e4e", height=10, aspect=2,
            linewidth = 3 )
ax2.fig.suptitle('Density after normalisation between [0,1]', size = 20)
plt.show(ax2)


# Step 4) Normalization by Sqrt Transform (For comparison)

# In[37]:


TrainDatacopy2 = TrainData.copy()
#TrainDataLogNorm = np.log(TrainDatacopy2)

#ax1 = sns.displot(TrainDataLogNorm, kind = "kde",color = "#e64e4e", height=10, aspect =2,
            #linewidth = 3, warn_singular=False )
#ax1.fig.suptitle('Distribution after Log transfomation', size = 20)
#plt.show(ax1)

#Some densities haven't been plotted since variance approaches or is equal to zero
#So to prepare for PCA dimensionality reduction, use squareroot transformation instead
TrainDatacopy3 = TrainData.copy()
TrainDataSqrtNorm = np.sqrt(TrainDatacopy3)

ax2 = sns.displot(TrainDataSqrtNorm, kind = "kde",color = "#e64e4e", height=10, aspect =2,
            linewidth = 3, warn_singular=False )
ax2.fig.suptitle('Distribution after Sqrt transfomation', size = 20)
plt.show(ax2)


# ### Outlier Detection

# In[38]:


# One method is Extreme Value Analysis, using interquartile range or z-score. This can only find outliers for numerical columns.
# It is useful for univariate outliers. We can use it for the columns with the widest set of values. So, for example, do not use for
# Credit_History, since there are only two values.

  
TrainData_XV = TrainData.copy()
Q1=TrainData_XV.quantile(0.25, numeric_only=True)
Q3=TrainData_XV.quantile(0.75, numeric_only=True)
IQR=Q3-Q1
TrainData_XV = TrainData_XV[~((TrainData_XV<(Q1-1.5*IQR)) | (TrainData_XV>(Q3+1.5*IQR)))]
print(TrainData_XV.shape) # No values are detected as outliers
print(TrainData.shape)


# In[39]:


# Another numerical approach is using standardized residuals from a multilinear regression model.
# Note: since we previously encoded our dataset, we can use this method.

from sklearn.linear_model import LinearRegression 

X= TrainData.iloc[:, :-1].values  
y= TrainData.iloc[:, -1].values 

reg = LinearRegression()
model = reg.fit(X, y)
print("regression coefficients: ", reg.coef_)
print("regression intercept: ", reg.intercept_)
print("model evaluation (r^2)", reg.score(X, y)) #R^2

predicted = model.predict(X)
residuals = list(np.subtract(y, predicted))
standardized_residuals = residuals / np.std(residuals) # |value| > 2 is outlier
# print("\n", y, "\n")
# print(predicted, "\n")
#print(standardized_residuals, "\n") # 13 outlier

TrainData_LR = TrainData.copy()
TrainData_LR['standardized_residuals'] = standardized_residuals



# removing outliers
TrainData_LR = TrainData_LR.drop(index=[row for row in TrainData_LR.index if 
                                        TrainData_LR.loc[row, 'standardized_residuals'] > 2
                                        or TrainData_LR.loc[row, 'standardized_residuals'] < -2])
# deleting last column
TrainData_LR = TrainData_LR.drop(columns=['standardized_residuals'])
TrainData_LR # 13 outlier


# In[40]:


# corr matrix (for visualization)
TrainData_LR.corr()


# In[41]:


# heatmap

font = {'family': 'serif',
        #'color':  'darkred',
        'weight': 'normal',
        'size': 7,
        }

sns.heatmap(
    TrainData_LR.corr(),         
    annot=True, annot_kws=font,
    vmin=-1, vmax=1) # more lightly colored = more highly correlated
plt.show()


# In[42]:


# Plotting LoanAmount, ApplicantIncome, Credit_History, and Loan_Status

#both numerical
sns.displot(
    data=TrainData_LR,
    #kind="kde", 
    x="LoanAmount", 
    y="ApplicantIncome", 
    #hue="Loan_Status",
    #col="Credit_History"
)
plt.show()
# both categorical
sns.displot(
    data=TrainData_LR,
    #kind="swarm", 
    x="Credit_History", 
    y="Loan_Status"
)
plt.show()


# In[43]:


# An approach that can be used for both multivariate and univariate analysis is Isolation Forest. 
# This approach splits the data and isolates the samples in the form of a tree.
# The number of splittings required to isolate an outlier is lower than normal data.
# This is a type of unsupervised anomaly detection

from sklearn.ensemble import IsolationForest

model_IF = IsolationForest(contamination=float(0.05), random_state=42) 
# try with contamination = 0.001, 0.1, 0.5 to change the number of outliers we want to eliminate
model_IF.fit(TrainData)

TrainData_IF = TrainData_LR.copy()
input_features = TrainData_IF.columns
TrainData_IF['anomaly_scores'] = model_IF.decision_function(TrainData_IF[input_features]) # outlier score. -ve -> outlier
TrainData_IF['anomaly'] = model_IF.predict(TrainData_IF[input_features]) # outlier:-1  inlier:1

TrainData_IF # 27 outlier
#print(model_IF.decision_function(TrainData_IF[input_features]))
#print(model_IF.predict(TrainData_IF[input_features]))


# In[44]:


# Which method to use? Linear Regression.

TrainData = TrainData_LR.copy()


# ### Handle Noisy Data

# #### Binning By pd.cut

# In[45]:


#Bin by Applicant Income, Coapplicant Income, LoanAmount, Loan_Status
print(TrainData['ApplicantIncome'].describe(), '\n')
#ApplicantIncomeCut = pd.cut(TrainData.iloc[:, 5], 4)
#print(ApplicantIncomeCut)
Bins = 10 #example number of bins
BinSize = (81000 - 150)/Bins
print(pd.cut(TrainData['ApplicantIncome'], Bins, precision = 0).value_counts(sort=False))


# In[46]:


print(TrainData['CoapplicantIncome'].describe(), '\n')
print(pd.cut(TrainData['CoapplicantIncome'], Bins, precision = 0).value_counts(sort=False))


# In[47]:


print(TrainData['LoanAmount'].describe(), '\n')
print(pd.cut(TrainData['LoanAmount'], Bins, precision = 0).value_counts(sort=False))


# In[48]:


print(TrainData['Loan_Status'].describe(), '\n')
print(pd.cut(TrainData['Loan_Status'], Bins, precision = 0).value_counts(sort=False))


# In[49]:


# When categories are harder to define, we will use binning methods.

#TODO move this to after Binning and (maybe) change up the binned vars
# sns.scatterplot( data=TrainData,
#     x="CoapplicantIncome", y="LoanAmount", hue="Loan_Status" #col="Loan_Status  ", style="smoker", size="size",
# )


# #### Binning By Feature Engine

# In[50]:


import numpy.typing
from feature_engine.discretisation import EqualWidthDiscretiser
#Bin by Applicant Income, Coapplicant Income, Loan Amount, Loan_Status
ApplicantIncomeFE = EqualWidthDiscretiser(bins=10, return_object = True, return_boundaries = True)
#ApplicantIncomeFE.fit(TrainData)
#ApplicantIncomeFE.transform(TrainData)["ApplicantIncome_b"].value_counts()
#ApplicantIncomeFE = EqualWidthDiscretiser()
#print(ApplicantIncomeFE)
ApplicantIncomeFE.fit(TrainData)
ApplicantIncomeFE.transform(TrainData)["ApplicantIncome"].value_counts()


# In[51]:


CoapplicantIncomeFE = EqualWidthDiscretiser(bins=10, return_object = True, return_boundaries = True)
CoapplicantIncomeFE.fit(TrainData)
CoapplicantIncomeFE.transform(TrainData)["CoapplicantIncome"].value_counts()


# In[52]:


LoanAmountFE = EqualWidthDiscretiser(bins=10, return_object = True, return_boundaries = True)
LoanAmountFE.fit(TrainData)
LoanAmountFE.transform(TrainData)["LoanAmount"].value_counts()


# In[53]:


# When categories are harder to define, we will use binning methods.

#TODO move this to after Binning and (maybe) change up the binned vars
# sns.scatterplot( data=TrainData,
#     x="CoapplicantIncome", y="LoanAmount", hue="Loan_Status" #col="Loan_Status  ", style="smoker", size="size",
# )


# #### Binning By KBinsDiscretizer Example

# In[54]:


from sklearn.preprocessing import KBinsDiscretizer


# In[55]:


#Default bins
TrainDataAmounts = TrainData[NumericData]
print(TrainDataAmounts)
TrainDataEqual = KBinsDiscretizer(n_bins = 10, strategy = 'uniform', encode = 'ordinal')
n = TrainDataEqual.fit(TrainDataAmounts)
print(n.bin_edges_)


# In[56]:


#Default bins #Number of bins has been decresed
TrainDataAmounts = TrainData[NumericData]
print(TrainDataAmounts)
TrainDataEqual = KBinsDiscretizer(n_bins = 2, strategy = 'quantile', encode = 'ordinal')
n = TrainDataEqual.fit(TrainDataAmounts)
print(n.bin_edges_)


# In[57]:


# When categories are harder to define, we will use binning methods.

#TODO move this to after Binning and (maybe) change up the binned vars
# sns.scatterplot( data=TrainData,
#     x="CoapplicantIncome", y="LoanAmount", hue="Loan_Status" #col="Loan_Status  ", style="smoker", size="size",
# )


# ### Data Discretization

# In[58]:


# Remove irrelevant data
TrainData.corr()
# TODO move this to after discretization, and check it with loan_status. if weakly correlated with loan_status, you can remove the feature
# TODO we can carry out dimensionality reduction


# ### Data Normalization

# #### Using Z-score

# In[59]:


# Analyze their impact, then decide whether to remove the outliers
# We can also use regression to get standardized_residuals and analyze the outliers


# ### Correlations Visualization

# In[60]:


corr_matrix = TrainData.corr()
print(corr_matrix)


# In[61]:


#the following is the visualization of the correlation matrix which shows the strength of the relation between each 2 variables
# you will find that we did not visualize the variable with itself because it will always be (1)


# # Dimensionality Reduction with PCA

# In[62]:


#Calculate variance of each normalised feature, by square root
#Last Column is the target
import statistics

variances = TrainDatacopy3.var()
print(variances, '\n')
featurenames = list(TrainDatacopy3.iloc[:, :-1])
#print(featurenames)
targets = TrainDatacopy3['Loan_Status']
#print(targets)
#New dataframe containing only 3 columns with highest variances as principal features, delta degrees of freedom = 1


# In[63]:


#PCA can be trained by various methods, two are discussed here: Logistic Regression and Random Forest
#Logistic Regression:predicts binary output
#Uses 1/ 1 + exp(-(bo + b1*x))
#Random Forest:uses multiple decision trees and combines their output to get a decision
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#Use all expect decision column, aka last column
X = TrainDataSqrtNorm.iloc[:,:-1]      
Y = TrainDataSqrtNorm.iloc[:,-1]
random_state = 42 #shuffling of executions of splitting, best 42
Xtrain, Xtest, Ytrain, Ytest =                     train_test_split(X, Y,
                     test_size = 0.2, #Test size: 80% training, 20% testing
                     shuffle = True,
                     random_state = random_state)
pca = PCA()
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
explainedvariance = pca.explained_variance_ratio_
print("Variance: ",explainedvariance)

LogisticR = LogisticRegression(max_iter = 5000)
LogisticR.fit(Xtrain, Ytrain)
#Get score after (forward) iterations
print("Logistic Regression Score:", LogisticR.score(Xtest,Ytest))


# In[64]:


#Use same splitting in the Random Forest algorithm
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth = 5, random_state = 0)
classifier.fit(Xtrain, Ytrain)

Ypredict = classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

Lossmatrix = confusion_matrix(Ytest, Ypredict)
print("Error matrix:", Lossmatrix)
print("Random Forest Score:", accuracy_score(Ytest, Ypredict))


# ## Regression: Phase 2
