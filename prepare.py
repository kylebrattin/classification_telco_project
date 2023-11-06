from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


# ------------------- TELCO DATA -------------------

def clean_telco(telco_df):
    '''USE THIS FOR INITIAL CLEAN - Cleans the telco dataset for exploring
    leaving the booleans and cleaning null values and dropping dup columns
    
    arguments: telco_df
    
    return: a clean dataframe ready to explore'''

    # drop duplicate columns and customer_id for view
    telco_df = telco_df.drop(columns =['payment_type_id','internet_service_type_id','contract_type_id', 'customer_id'])
    # fill nulls and change total_charges to float
    telco_df.total_charges = telco_df.total_charges.str.replace(' ', '0').astype(float)
    # drops (automatic) from the payment_type
    telco_df['payment_type'] = telco_df['payment_type'].str.replace(' (automatic)', '')

    return telco_df


def split_telco_data(df, target):
    '''
    split telco data into train, validate, test

    returns train, validate, test
    '''

    train_val, test = train_test_split(df,
                                   train_size=0.8,
                                   random_state=1108,
                                   stratify=df[target])
    train, validate = train_test_split(train_val,
                                   train_size=0.7,
                                   random_state=1108,
                                   stratify=train_val[target])
    
    print(f'Train: {len(train)/len(df)}')
    print(f'Validate: {len(validate)/len(df)}')
    print(f'Test: {len(test)/len(df)}')
    

    return train, validate, test


def prep_telco(df_telco):
    '''USE BEFORE MODELING - Preps the telco dataset for ml modeling by dropping columns
    creating dummy values, and replacing null values in total_charges, keeps customer_id
    '''

  
    #drop duplicate columns
    df_telco = df_telco.drop(columns =['payment_type_id','internet_service_type_id','contract_type_id', 'gender', 'partner', 'dependents',
                                       'phone_service', 'multiple_lines', 'paperless_billing', 'streaming_tv', 'streaming_movies'])
    #create dummies
    dummy_list = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'churn', 'contract_type',
                  'internet_service_type','payment_type']
    dummy_df = pd.get_dummies(df_telco[dummy_list], dtype=int, drop_first=True)
    # concat dummy & telco_df
    df_telco = pd.concat([df_telco, dummy_df], axis=1)
    # drop str column categories
    cols_to_drop = ['online_security', 'online_backup', 'device_protection', 'tech_support', 
                    'churn', 'contract_type', 'internet_service_type', 'payment_type']
    df_telco = df_telco.drop(columns= cols_to_drop)
    #total_charges.str.replace(' ', '0').astype(float)
    df_telco.total_charges = df_telco.total_charges.str.replace(' ', '0').astype(float)
    

    return df_telco



def next_split(train, validate, test):
    '''This function creates your modeling variables with the train, validate, test 
    sets and returns them
    
    '''

    X_train = train.drop(columns=['churn_Yes','customer_id', 'senior_citizen'])

    X_validate = validate.drop(columns=['churn_Yes','customer_id', 'senior_citizen'])

    X_test = test.drop(columns=['churn_Yes','customer_id', 'senior_citizen'])

    y_train = train['churn_Yes']

    y_validate = validate['churn_Yes'] 

    y_test = test['churn_Yes']

    return X_train, X_validate, X_test, y_train, y_validate, y_test