import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
import datetime as dt
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
    df.last_pymnt_d=pd.to_datetime(df.last_pymnt_d)
    df.issue_d=pd.to_datetime(df.issue_d)
    
    return df

