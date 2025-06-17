#!/usr/bin/env python
# coding: utf-8

# In[111]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import normaltest
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import statsmodels.api as s1
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

datafile = pd.read_csv('C:/Users/siddh/OneDrive/Desktop/Python advanced/Patient details and baseline weight.csv')

bmivalue = (datafile["Baseline Weight lbs"] / (datafile["Height inch"] ** 2))*703

datafile["BMI"]=bmivalue.round(0)
datafile.drop_duplicates(inplace=True)
for i in range(len(datafile)):
    if datafile["BMI"][i]>=18.5 and datafile["BMI"][i]<=24:
        datafile["BMI desc"][i]= b"Normal"
    if datafile["BMI"][i]>=25 and datafile["BMI"][i]<=29:
        datafile["BMI desc"][i]= b"Overweight"
    if datafile["BMI"][i]>=30:
        datafile["BMI desc"][i]= b"Obese"
    
#Calculating the BMR value and appending to table
datafile['BMR']=''
for i in datafile.index:
    if datafile['Gender'][i] == 'male':
        datafile['BMR'][i] = 66.47 + (6.24 * datafile["Baseline Weight lbs"][i]) + (12.7 * datafile["Height inch"][i]) - (6.755 * datafile["Age"][i])
    elif datafile['Gender'][i] == 'female':
        datafile['BMR'][i] = 655.1 + (4.35 * datafile["Baseline Weight lbs"][i]) + (4.7 * datafile["Height inch"][i]) - (4.7 * datafile["Age"][i])

print(datafile)

print()
print("Comorbidities of patient")
#The table with patient comorbidity details
datafile1 = pd.read_csv('C:/Users/siddh/OneDrive/Desktop/Python advanced/Comorbidity patient.csv')
datafile1=datafile1.drop_duplicates()
print(datafile1)

print()
print("Post Wellness Program patient weight and body fat measurements")
datafile2 = pd.read_csv('C:/Users/siddh/OneDrive/Desktop/Python advanced/Post treatment weight.csv')
datafile2=datafile2.drop_duplicates()
print(datafile2)

print()
print("Post Wellness Program patient waist circumference measurements")
datafile3 = pd.read_csv('C:/Users/siddh/OneDrive/Desktop/Python advanced/Waistcircumference.csv')
datafile3=datafile3.drop_duplicates()
print(datafile3)

#Calculating the weightloss and bodyfat change from baseline to 52 weeks of program
datafile2['Weightloss']=''
datafile2['Bodyfat_diff']=''
for i in datafile2.index:
    datafile2["Weightloss"][i]= datafile2['Baseline Weight lbs'][i]-datafile2['Wtwk52'][i]
    datafile2["Bodyfat_diff"][i]= datafile2['P_baselinebodyfat%'][i]-datafile2['P_bodyfat52week'][i]
    
print(datafile2)

#Calculating the waist circumference change from baseline to 52 weeks of program
print()
print("Waist circumference difference between baseline and 52 weeks of weightloss program")
datafile3['Waist_diff']=''
for i in datafile3.index:
    datafile3["Waist_diff"][i]= datafile3['Wstbase'][i]-datafile3['Wst52'][i]
print(datafile3)

print()
print("Merging table 1 and 2")
final1=pd.merge(datafile,datafile1,how='inner',on='PatientID',)
print("Merging table 1, 2,3")
final2=pd.merge(final1,datafile2,how='inner',on='PatientID')
print("Merging all tables")
final3=pd.merge(final2,datafile3,how='inner',on='PatientID')
print(final3)

print(final3.dtypes)
#converting object datatype to float
final3['Weightloss'] = pd.to_numeric(final3['Weightloss'])
final3['Bodyfat_diff']=pd.to_numeric(final3['Bodyfat_diff'])
final3['Waist_diff']=pd.to_numeric(final3['Waist_diff'])
final3['BMR']=pd.to_numeric(final3['BMR'])
print(final3.dtypes)

print()
#Finding normality for baseline weight and post treatment weight
print('Normality test for baseline weight')
result=normaltest(datafile2['Baseline Weight lbs'])
print(result)
if result.pvalue>0.05:
    print("Data is normally distributed")
else:
    ("Data is not normally distributed")

print('Normality test for post treatment weight')
result1=normaltest(datafile2['Wtwk52'])
print(result1)
if result1.pvalue>0.05:
    print("Data is normally distributed")
else:
    ("Data is not normally distributed")
    
print()
print("Mean,median,mode for weighloss")
mean = final3['Weightloss'].mean()
median = final3['Weightloss'].median()
mode = final3['Weightloss'].mode().iloc[0]

print("Mean:",mean)
print("Median:",median)
print("Mode:",mode)

print("Standard Deviation and Variance for weighloss")
std_dev = final3['Weightloss'].std()
print("Standard Deviation:",std_dev)
variance = final3['Weightloss'].var()
print("Variance:",variance)

print()
print("Min and max values in BMI")
min_value = final3['BMI'].min()
print("Minimum BMI value recorded is:",min_value)
max_value = final3['BMI'].max()
print("Maximum BMI value recorded is:",max_value)

print()
print("Skew and Kurtosis for waist circumference change in one year")
print("Skew is:",final3['Waist_diff'].skew())
print("Kurtosis is:",final3['Waist_diff'].kurtosis())

print()
print("Two sample t-test")
print("Null:Weightloss is same for male and female.")
print("Alternate:Weightloss is not same for male and female.")
Wt_F=final3[final3['Gender']=='female']['Weightloss']
Wt_M=final3[final3['Gender']=='male']['Weightloss']
result2= ttest_ind(Wt_F,Wt_M)
print(result2)
alpha=0.05
if result2.pvalue<alpha:
    print("Result: Accept alternate hypothesis. Weightloss is not same for male and female.")
else:
    print("Result: Accept null hypothesis. Weightloss is same for male and female.")


print()
print("Two way ANOVA")
print("Checking the effect of Gender and smoking status on weightloss and their interaction on weighloss")
model=ols('Weightloss ~ C(P_smoking)+C(Gender)+C(P_smoking):C(Gender)',data=final3).fit()
result3=s1.stats.anova_lm(model,typ=2)
print(result3)

alpha = 0.05
print()
if result3['PR(>F)'][0] < alpha:
    print("Result: Accept alternate hypothesis. There is a significant effect of smoking on Weightloss.")
else:
    print("Result: Accept null hypothesis. There is no significant effect of smoking on Weightloss.")

if result3['PR(>F)'][1] < alpha:
    print("Result: Accept alternate hypothesis. There is a significant effect of Gender on Weightloss.")
else:
    print("Result: Accept null hypothesis. There is no significant effect of Gender on Weightloss.")

if result3['PR(>F)'][2] < alpha:
    print("Result: Accept alternate hypothesis: There is a significant interaction effect between smoking and Gender on Weightloss.")
else:
    print("Result: Accept null hypothesis: There is no significant interaction effect between smoking and Gender on Weightloss.")

print()
print("Correlation")
print("Finding the correlation between Systolic BP and Weightloss")
weightloss=final3['Weightloss']
sys=final3['P_sysBP']
corr_coeff, p_value = stats.pearsonr(weightloss,sys)
print("Correlation coefficient is:",corr_coeff)
print("pvalue is:", p_value)
print("Result:")
if corr_coeff<0:
    print("There is a negative correlation between weightloss and systolic BP.")
else:
    print("There is a positive correlation between weightloss and systolic BP.")

if p_value<0.05:
    print("There is a statistically significant correlation between weightloss and systolic BP")
else:
    print("There is not a statistically significant correlation between weightloss and systolic BP")

print()
print("Linear regression")
X = final3[['Weightloss']]
Y = final3['P_sysBP']
X = s1.add_constant(X)
model = s1.OLS(Y,X).fit()
print(model.summary())

print()
print("Visualization")
#pie chart for BMI descriptions
import pandas as pd
normal_count= final3[final3['BMI desc']== b'Normal'].shape[0]
obese_count = final3[final3['BMI desc'] == b'Obese'].shape[0]
overweight_count = final3[final3['BMI desc'] == b'Overweight'].shape[0]

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
BMI_category=['Normal','Overweight','Obese']
labels=['24=<Normal>=18.5','29<=Overweight>=25','Obese>=30']
people=[normal_count,overweight_count,obese_count]
ax.pie(people,labels=BMI_category,autopct='%1.1f%%')
plt.title("People in the program and their BMIs")
plt.legend(loc='lower center', ncols=3,bbox_to_anchor=(0.5,-0.1),labels=labels)
plt.show()
print()

#Line plot to show baseline weight and weight after 52 weeks of intervention
fig=plt.figure()
participants=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
pre_weights=final3['Baseline Weight lbs_x']
post_weights=final3['Wtwk52']
plt.plot(participants, pre_weights, marker='*', linestyle='-', color='b', label='Pre-intervention Weight')
plt.plot(participants, post_weights, marker='*', linestyle='-', color='g', label='Post-intervention Weight')
plt.title('Pre and Post-Weightloss Program Weights')
plt.xlabel('Participant')
plt.ylabel('Weight (lbs)')
plt.legend()
plt.xticks(participants)
plt.show()

#Histogram for BMI showing how BMI varies for all particpants
age=final3['BMI']
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_title('BMI histogram')
ax.set_xticks([0,5,10,15,20,25,30,35,40,45,50])
ax.set_xlabel('BMI')
ax.set_ylabel('Participants')
ax.hist(age)
plt.show()

#Bar chart to show the participant count for comorbidities
fig,ax=plt.subplots()
labels=['Alcohol','Smoking','Diabetes','Drugs']
yes=[9,8,8,7]
no=[11,12,12,13]
width=0.4
x=np.arange(len(labels))
ax.bar(x-width/2,yes,width,label='Yes',color='r')
ax.bar(x+width/2,no,width,label='No',color='b')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of participants')
ax.legend()
plt.show()
print()

#calculations for report
gendercount=final3['Gender'].value_counts()
men_count=gendercount.get('male',0)
fem_count=gendercount.get('female',0)

counts = final3['P_smoking'].value_counts()
sm_yes = counts.get('yes', 0)
sm_no = counts.get('no', 0)
#print("Count of 'yes':", sm_yes)
#print("Count of 'no':", sm_no)

counts = final3['P_alcohol'].value_counts()
al_yes = counts.get('yes', 0)
al_no = counts.get('no', 0)
#print("Count of 'yes':", al_yes)
#print("Count of 'no':", al_no)

counts = final3['P_drugs'].value_counts()
dr_yes = counts.get('yes', 0)
dr_no = counts.get('no', 0)
#print("Count of 'yes':", dr_yes)
#print("Count of 'no':", dr_no)

counts = final3['P_diabetes'].value_counts()
d_yes = counts.get('yes', 0)
d_no = counts.get('no', 0)
#print("Count of 'yes':", d_yes)
#print("Count of 'no':", d_no)

positive_count = (final3['Weightloss'] > 0).sum()
negative_count = (final3['Weightloss'] < 0).sum()

bodyfat_dec = (final3['Bodyfat_diff'] > 0).sum()
bodyfat_inc = (final3['Bodyfat_diff'] < 0).sum()

#REPORT
print()
print()
print("-"*60)
print("-"*60)
print()
print("REPORT")
print("Weightloss Program- Pre and Post treatment evaluations")
print()
print("-"*60)
print("-"*60)
print()
print("How many participants have reduced weight after program?",positive_count)
print("How many participants have gained weight after program?",negative_count)
print()
print("How many participants have gained bodyfat after program?",bodyfat_inc)
print("How many participants have reduced bodyfat after program?",bodyfat_dec)
print()
print("How many men in the dataset?",men_count)
print("How many women in the dataset?",fem_count)
print()
print("How many individuals have said yes to smoking?",sm_yes)
print("How many individuals have said no to smoking?",sm_no)
print("How many individuals have said yes to alcohol?",al_yes)
print("How many individuals have said no to alcohol?",al_no)
print("-"*60)
print("-"*60)
print("Created By:")
print("Sindhuri Chintakunta, Graduate student")
print("Rutgers University")
print("-"*60)
print("-"*60)
print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




