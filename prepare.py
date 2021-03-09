import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def clean_telco(df):
    '''
    clean_telco will take one argument df, a pandas dataframe, anticipated to be the telco_churn dataset
    and will change the monthly_charges and total_charges columns to float and will encode and remove specified columns
    return: a single pandas dataframe with the above operations performed
    '''
    df.total_charges.replace(to_replace=' ',value=np.nan,inplace=True)
    df['total_charges'] = df.total_charges.astype(float).fillna(df.monthly_charges)
        
    
    df.drop(columns=['payment_type_id', 'internet_service_type_id',
                     'contract_type_id'], inplace=True)

    dummy_df = pd.get_dummies(df[['gender',
                                  'partner',
                                  'dependents',
                                  'phone_service',
                                  'paperless_billing',
                                  'churn'
                                  ]], drop_first=True)

    dummy_df2 = pd.get_dummies(df[['online_security',
                                   'online_backup',
                                   'device_protection',
                                   'tech_support',
                                   'streaming_tv',
                                   'streaming_movies',
                                   'multiple_lines',
                                   'contract_type',
                                   'internet_service_type',
                                   'payment_type']])

    dummy_df2.drop(columns=['online_security_No internet service',
                            'online_backup_No internet service',
                            'device_protection_No internet service',
                            'tech_support_No internet service',
                            'streaming_tv_No internet service',
                            'streaming_movies_No internet service',
                            'multiple_lines_No phone service',
                           
                            ], inplace=True)

    encode_df = pd.concat([df, dummy_df, dummy_df2], axis=1)

    encode_df.drop(columns=['gender',
                            'partner',
                            'dependents',
                            'phone_service',
                            'multiple_lines',
                            'online_security',
                            'online_backup',
                            'device_protection',
                            'tech_support',
                            'streaming_tv',
                            'streaming_movies',
                            'paperless_billing',
                            'churn',
                            'contract_type',
                            'internet_service_type',
                            'payment_type',
                            'online_security_No',
                            'online_backup_No',
                            'device_protection_No',
                            'tech_support_No',
                            'streaming_tv_No',
                            'streaming_movies_No',
                            'multiple_lines_No',                            
                            ], inplace=True)
    
    encode_df.columns = ['customer_id',
                 'senior_citizen',
                 'tenure',
                 'monthly_charges',
                 'total_charges',
                 'gender_Male',
                 'partner',
                 'dependents',
                 'phone_service',
                 'paperless_billing',
                 'churn',
                 'online_security',
                 'online_backup',
                 'device_protection',
                 'tech_support',
                 'streaming_tv',
                 'streaming_movies',
                 'multiple_lines',
                 'Month_to_month',
                 'One_year_contract',
                 'Two_year_contract',
                 'DSL',
                 'Fiber_optic',
                 'No_internet',
                 'Bank_transfer_(automatic)',
                 'Credit_card_(automatic)',
                 'Electronic_check',
                 'Mailed_check']
    
    return encode_df

def generic_split(df, stratify_by=None):
    """
    Crude train, validate, test splits
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(df, test_size=.3, random_state=123)
    else:
        train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate[stratify_by])
    s
    return train, validate, test

def get_metrics(model, X, y):
    '''
    get_metrics_bin will take in a sklearn classifier model, an X and a y variable and utilize
    the model to make a prediction and then gather accuracy, class report evaluations
    return:  a classification report as a pandas DataFrame
    '''
    y_pred = model.predict(X)
    accuracy = model.score(X, y)
    conf = confusion_matrix(y, y_pred)
    print('confusion matrix: \n', conf)
    print()
    class_report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).T
    tpr = conf[1][1] / conf[1].sum()
    fpr = conf[0][1] / conf[0].sum()
    tnr = conf[0][0] / conf[0].sum()
    fnr = conf[1][0] / conf[1].sum()
    print(f'''
    The accuracy for our model is {accuracy:.4}
    The True Positive Rate is {tpr:.3}, The False Positive Rate is {fpr:.3},
    The True Negative Rate is {tnr:.3}, and the False Negative Rate is {fnr:.3}
    ''')
    return class_report

    import pandas as pd
import numpy as np
import os

### from acquire.py

from env import host, user, password
from pydataset import data
from acquire import get_connection, new_telco_data, get_telco_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



# clean data followed by creating train/validate/test function

def clean_telco(df):
    '''
    clean_telco will take a dataframe acquired as df and remove columns that are:
    duplicates,
    have too many nulls,
    and will fill in the missing values in total_charges as 0 since those customers are in the first month of  contract
    We will be encoding gender, paperless billing, contract type, internet type, streaming tv, movies, paperless billing, contract type, and payment type.
    
    return: single cleaned dataframe
    '''
    df.drop_duplicates(inplace=True)
    
     # Converting the total charges column to a numeric type from object
    df["total_charges"] = pd.to_numeric(df.total_charges, errors='coerce')
    
    
    # Fill NaN values in total_charges column with 0
    df['total_charges'] = df['total_charges'].fillna(value=0)
    
    
     # create new average monthly charges column
    df['average_charges'] = round((df['total_charges']/df['tenure']), 2)
    
    
    # Fill NaN values in average_charges column with 0
    df['average_charges'] = df['average_charges'].fillna(value=0)
    
    
    # create a tenure in years column
    df['tenure_years'] = round(df.tenure / 12, 2)
    
    
    
     # Encode churn in one column to use when 'yes' 'no' can't be.
    df['encoded_churn'] = df['churn'].map( 
                   {'Yes':1 ,'No':0})
    
     # create new column for customer who have no partner and no dependents    
    df['no_partner_depend'] = (df['partner'] == 'No') & (df['dependents'] == 'No')
    
    
    # encode above boolean column into 0 or 1
    df.no_partner_depend = df.no_partner_depend.replace({True: 1, False: 0})
    
    
    # phone_service and multiple_lines
    df['phone_lines'] = (df['phone_service'] == 'Yes') & (df['multiple_lines'] == 'Yes')
    
    
    # encode above boolean column into 0 or 1
    df.phone_lines = df.phone_lines.replace({True: 1, False: 0})
    
    
    # create new column for customer who have streaming_tv & streaming_movies
    df['stream_tv_mov'] = (df['streaming_tv'] == 'Yes') & (df['streaming_movies'] == 'Yes')
    
    
    # encode above boolean column into 0 or 1
    df.stream_tv_mov = df.stream_tv_mov.replace({True: 1, False: 0})
    
    
    # create new column for customer who have online_security & online_backup
    df['online_sec_bkup'] = (df['online_security'] == 'Yes') & (df['online_backup'] == 'Yes')
    
    # encode above boolean column into 0 or 1

    df.online_sec_bkup = df.online_sec_bkup.replace({True: 1, False: 0})
    
    
     # create dummy columns of encoded categorical variables
    dummies = pd.get_dummies(df[['gender', 'partner', 'dependents', 'device_protection','tech_support', 'paperless_billing', 'contract_type', 'internet_service_type', 'payment_type']], drop_first=False)
   


    # create a dropcols where all columns that were created into dummies will be dropped
    dropcols = ['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'multiple_lines', 'gender', 'partner', 'dependents', 'phone_service', 'device_protection','online_security', 'online_backup', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'contract_type', 'internet_service_type', 'payment_type']
    
    
    # drop cols from above
    df.drop(columns=dropcols, inplace=True)
    
    # combine the original data frame with the new dummies columns
    df = pd.concat([df, dummies], axis=1)
    
    
    # rename columns    
    df.columns = ['customer_id',
 'senior_citizen',
 'tenure_in_months',
 'monthly_charges',
 'total_charges',
 'churn',
 'average_charges',
 'tenure_in_years', 
 'encoded_churn',
 'no_partner_depend',
 'phone_lines',
 'stream_tv_mov',
 'online_sec_bkup',
 'female',
 'male',
 'no_partner',
 'has_partner',
 'dependents_no',
 'dependents_yes',
 'device_protection_no',
 'device_protection_no_int',
 'device_protection_yes',
 'tch_support_no',
 'tch_support_no_int',
 'tch_support_yes',
 'paperless_billing_no',
 'paperless_billing_yes',
 'monthly_contract',
 'one_yr_contract',
 'two_yr_contract',
 'has_dsl',
 'has_fiber_optic',
 'no_internet',
 'pmt_bank transfer',
 'pmt_cc',
 'pmt_electronic_check',
 'pmt_mailed_check']

    

    return df


def train_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.churn
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.churn,
    )
    return train, validate, test