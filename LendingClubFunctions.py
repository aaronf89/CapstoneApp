import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
import datetime as dt
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, grid_search
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestClassifier
plt.style.use('ggplot')



#Import data function
def GetData(cols, files):

    frames=[]
    for file in files:
        frames.append(pd.read_csv(file, usecols=cols))
    
    return pd.concat(frames, ignore_index=True)


#Functions to clean up data
def ConvPercent(x):
    return float(str(x).strip('%').strip(' '))/100.0
    
def ConvertEmpYears(x):
    #print x
    if str(x)=='nan':
        return np.nan
    else:
        y=str(x).replace('+', ' ').replace('<',' ').split(' ')
        if len(y[0])==0:
            return 0
        elif y[0]=='n/a' or y[0]=='na':
            return np.nan
        else:
            return int(y[0])

        
def ConvertOwnership(x):
    
    if x=='RENT':
        return 0.0
    elif x=='MORTGAGE':
        return 1.0
    elif x=='OWN':
        return 2.0
    else:
        return np.nan
        
def ConvHist(x):
    
    return 12.*(2015-x.year)+11-x.month

def ConvTerm(x):

    return float(str(x).strip().split(' ')[0])


def ConvCredit(x):
    if ":" in str(x):
        return str(x).split(':')[1]
    else:
        return str(x)

def ConvLoanStatus(x, Trouble=[], Success=[]):

    if x in Trouble:
        return bin(0)
    elif x in Success:
        return bin(1)    
    else:
        return 2
        
def IntQuartiles(x, q1=.25, q2=.5, q3=.75):
    
    if x<q1:
        return 'Tier 1'
    elif x>q1 and x<q2:
        return 'Tier 2'
    elif x>q2 and x<q3:
        return 'Tier 3'
    else:
        return 'Tier 4'
    
#Summary of defaulted loans
def GetSummary(df):

    return pd.Series([100.*df.groupby('loan_status').count().loan_amnt.ix['0b0']/float(df.loan_amnt.count()),
           df.funded_amnt_inv.sum()/10**9, df.groupby('loan_status').sum().Loss.ix['0b0']/10**6], 
           index=['Default Percentage', 'Total Funded Loans (billions)', 'Total Lost Principal (millions)'])


#Clean Data
def CleanData(df, Trouble=[], Success=[]):
    df['revol_util']=(df['revol_util'].apply(ConvPercent))
    df['emp_length']=(df['emp_length'].apply(ConvertEmpYears))
    df['OldHome']=df.home_ownership
    df['home_ownership']=(df['home_ownership'].apply(ConvertOwnership))
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['earliest_cr_line'] = df['earliest_cr_line'].apply(ConvHist)
    df['term'] = df['term'].apply(ConvTerm)
    df['int_rate']=(df['int_rate'].apply(ConvPercent))
    df['OldStatus']=df.loan_status.apply(ConvCredit)
    df['loan_status']=df.loan_status.apply(ConvLoanStatus, Trouble=Trouble, Success=Success)
    
    temp1=df.purpose.unique().tolist()
    temp1.sort()
    purposedict={p:n for p,n in zip(temp1, range(len(temp1)))}
    purposedictrev={n:p for p,n in purposedict.items()}
    df['purposedummy']=df.purpose.apply(lambda x: purposedict[x])
    
    temp1=df.addr_state.unique().tolist()
    temp1.sort()
    addr_statedict={p:n for p,n in zip(temp1, range(len(temp1)))}
    addr_statedictrev={n:p for p,n in addr_statedict.items()}
    df['addr_statedummy']=df.addr_state.apply(lambda x: addr_statedict[x])
    
    temp1=df.sub_grade.unique().tolist()
    temp1.sort()
    sub_gradedict={p:n for p,n in zip(temp1, range(len(temp1)))}
    sub_gradedictrev={n:p for p,n in sub_gradedict.items()}
    df['sub_gradedummy']=df.sub_grade.apply(lambda x: sub_gradedict[x])
    
    df['Fraction_Of_Total']=df.total_pymnt/(df.installment*df.term)
    
    #Now sort interest rate into quatiles
    Q1,Q2,Q3=(df['int_rate'].describe()).ix[[4, 5,6]]
    df['IntCat']=df.int_rate.apply(IntQuartiles, q1=Q1,q2=Q2,q3=Q3)
    
    #Find the percent lost by any given loan (only relevant to defaulted loans)
    #df['Loss']=(df.funded_amnt_inv-df.total_pymnt_inv)
    #df['PerLoss']=(df.funded_amnt_inv-df.total_pymnt_inv)/df.funded_amnt_inv
    df['ROI']=(df.total_pymnt-df.funded_amnt)/df.funded_amnt
    df['ROIAnnualized']=(1+df['ROI'])**(12./df['term'])-1
    df['Fraction_Of_Total']=df.total_pymnt/(df.installment*df.term)
    df['PerLoss']=(df.total_pymnt-df.funded_amnt)/df.funded_amnt
    df['PerLossTerm']=(df.total_pymnt-df.installment*df.term)/(df.installment*df.term)
    
    return df

#Plot the distribution of lost principal
def PlotLostPrin(df):

    plt.figure(1)
    plt.clf()
    ax1=plt.hist((df[(df['loan_status']==bin(0)) & (df.funded_amnt_inv>1)].PerLoss).as_matrix()*100, bins=50,  color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=1)
    plt.xlabel('Fraction of Initial Loan Amount Lost (in percent)', fontsize=20)
    plt.ylabel('Frequency of Loss', fontsize=20)
    plt.title('Principal Lost When Borrower Fails to Pay', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


#Function to plot default rates among different home owners
def PlotHomeOwnership(df, ax):
    
    group1=df.groupby(['loan_status','OldHome']).count().loan_amnt
    totals=df.groupby(['OldHome']).count().loan_amnt
    Lost=100.*group1.ix['0b0']/totals
    ax=Lost[['RENT', 'OWN', 'MORTGAGE']].plot(kind='bar',  color='burlywood', fontsize=15, ax=ax)
    ax.set_xlabel('Living Situation')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Home Ownership')
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.title.set_fontsize(20)

    
#Function to plot default rates by credit length
def PlotCreditLength(df, ax):
    
    group1=df.groupby(['loan_status', pd.cut(df.earliest_cr_line, 6)])
    total=df.groupby([ pd.cut(df.earliest_cr_line, 6)])
    
    
    Lost=100.0*group1.count().loan_amnt.ix['0b0']/total.count().loan_amnt
    Lost.index=total.mean().earliest_cr_line.apply(round)
    ax=Lost.plot(kind='bar',  color='burlywood', fontsize=15, ax=ax)
    ax.set_xlabel('Months of Credit History')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Credit Length')
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.title.set_fontsize(20)
    
def PlotIntCat(df, ax):
    
    InterestRate=df.groupby('IntCat').mean().int_rate*100.
    DefaultRate=100.*df.groupby(['loan_status','IntCat']).count().int_rate.ix['0b0']/df.groupby('IntCat').count().int_rate
    Loss=df.groupby(['loan_status','IntCat']).mean().Loss.ix['0b0']
    

    ax=DefaultRate.plot(kind='bar', color='burlywood', label='Charged off/in default', fontsize=15, ax=ax)
    ax.set_xlabel('Interest Rate Category')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Interest Rate')
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.title.set_fontsize(20)
      
    return pd.DataFrame([InterestRate , DefaultRate , Loss ], index=['AveIntRate', 'DefaultRate', 'AveLoss']).T    
    
#Function that plots the portion of defaulted and successful loans in each interest rate quartile
def PlotFractions(df):
    grouped=df.groupby(['loan_status', 'IntCat'])
    PaidRates=(grouped.count()['int_rate'])['0b1']/((grouped.count()['int_rate'])['0b1'].sum())
    LostRates=(grouped.count()['int_rate'])['0b0']/((grouped.count()['int_rate'])['0b0'].sum())

    plt.figure(1)
    plt.clf()
    ax4=LostRates.plot(kind='bar', color='crimson', position=0, width=.25, label='Charged off/in default', fontsize=15)
    ax5=PaidRates.plot(kind='bar', color='burlywood', position=1, width=.25, label= 'Paid in full', fontsize=15)
    ax4.set_xlabel('Interest Rate Category')
    ax4.set_ylabel('Fraction of Loans')
    ax4.set_title('Fraction of Loans in Each Category')
    plt.legend( loc='best', prop={'size':15})
    ax4.xaxis.label.set_fontsize(20)
    ax4.yaxis.label.set_fontsize(20)
    ax4.title.set_fontsize(20)
    plt.show()
    