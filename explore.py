from scipy import stats
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def churn_overall(df):
    '''plots overall churn responses
    of the data frame and returns 
    a bargraph'''
    # create a figure
    fig = plt.figure(figsize=(12, 6)) 
    ax = fig.add_subplot(111)

    # proportion of observation of each class
    prop_response = df['churn'].value_counts(normalize=True)

    # create a bar plot showing the percentage of churn
    prop_response.plot(kind='bar', 
                    ax=ax)


    # set title and labels
    ax.set_title('Proportion of observations of the response variable',
                fontsize=18)
    ax.set_xlabel('churn',
                fontsize=14)
    ax.set_ylabel('proportion of observations',
                fontsize=14)
    ax.tick_params(rotation='auto')
    
def column_split(df):
    '''Takes the qualitative and quantitative columns and splits them
    as such. returns cat_cols, num_cols'''

    # separating our numeric and categorical columns:
    cat_cols, num_cols = [], []
    # set up a for loop to build those lists out:
    # so for every column in explore_columns:
    for col in df:
        # check to see if its an object type,
        # toss in categorical
        if df[col].dtype == 'O':
            cat_cols.append(col)
        # if its numeric:
        else:
            # check to see if we have more than just a few values:
            # if thats the case, toss it in categorical
            if df[col].nunique() < 10:
                cat_cols.append(col)
            # otherwise call it continuous by elim
            else:
                num_cols.append(col)

    return cat_cols, num_cols


def stack_plot(col_to_stack, df):
    '''Takes the prepared columns from column_split function and plots
    stacked percentage graphs of each category.
    returns visual barcharts'''
    
    for index, column in enumerate(col_to_stack):
        bar_by_cat = pd.crosstab(df[column], df['churn']).apply(lambda x: x/x.sum()*100, axis=1)
        bar_by_cat.plot(kind='bar', stacked=True)
        plt.ylabel('Percentage')
        plt.xlabel(column)

def get_chi_osec(df):
    '''gets the results of chi-square for churn and online_security'''
    on_sec = df.online_security[df.online_security != 'No internet service']
    observed = pd.crosstab(df.churn, on_sec)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_chi_backup(df):
    '''gets the results of chi-square for churn and online_backups'''
    on_back = df.online_backup[df.online_backup != 'No internet service']
    observed = pd.crosstab(df.churn, on_back)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_chi_dvpro(df):
    '''gets the  result of chi-square for churn and device_protections'''
    on_dev = df.device_protection[df.device_protection != 'No internet service']
    observed = pd.crosstab(df.churn, on_dev)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_chi_techsup(df):
    '''gets the result of chi-square for churn and tech_support'''
    on_tech = df.tech_support[df.tech_support != 'No internet service']
    observed = pd.crosstab(df.churn, on_tech)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_chi_contype(df):
    '''gets the result of chi-square for churn and contract type'''
    
    observed = pd.crosstab(df.churn, df.contract_type)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')