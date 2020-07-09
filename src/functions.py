import os
import pandas as pd
import numpy as np



def load_train_add_target_df():
    '''This function loads the three
    datasets from the data folder and
    creates a dataframe for each dataset.
    The train data and train target 
    datasets are merged together to 
    complete the training dataset with
    the target label'''
    train_data = pd.read_csv('../data/train_data.csv')
    train_target = pd.read_csv('../data/train_targets.csv')
    train = train_data.merge(train_target, on='id', how='inner')
    return train


def load_test_df():
    test = pd.read_csv('../data/test_set_values.csv')
    return test


def numeric_status_group():
    status_group_numeric = {'functional': 2,
                        'functional needs repair': 1,
                        'non functional': 0}
    return status_group_numeric


def encode_region(df, num_of_bins=7):
    '''
    Takes in Tanzania Water Point Data, groups by region,
    sorts regions by proportion of non-functional wells,
    and bins them in equally-sized bins according to
    num_of_bins parameter
    
    
    returns: DataFrame with 'region_bins' column added
             and with 'region' and 'region_code' columns dropped
    '''
    
    #group DataFrame by region and count each type of waterpoint
    reg = df.groupby('region')['status_group'].value_counts().unstack()

    #calculate proportion of non-functional waterpoints in each region
    reg['total'] = reg.sum(axis=1)
    reg['non'] = reg['non functional'] / reg['total']

    #sort by that proportion
    reg = reg.sort_values('non')

    #sort regions into specified number of equally wide bins
    bin_labels = list(range(num_of_bins))
    reg['region_bins'] = pd.cut(reg.non, bins=num_of_bins,
                          labels=bin_labels)
    codes = reg.region_bins
        
    #return bin numbers attached to dataframe
    return df.join(codes, on='region').drop(['region','region_code'], axis=1)


def categorize_funder(train):
    '''This function will go through every row in
    the dataframe column funder and if the value
    is any of the top 7 fields, will return those
    values. If the value does not equal any of these
    top 7 fields, it will return other.'''
    if train['funder'] == 'Government Of Tanzania':
        return 'govt'
    elif train['funder'] == 'Danida':
        return 'danida'
    elif train['funder'] == 'Hesawa':
        return 'hesawa'
    elif train['funder'] == 'Rwssp':
        return 'rwssp'
    elif train['funder'] == 'World Bank':
        return 'world_bank'
    elif train['funder'] == 'Kkkt':
        return 'kkkt'
    elif train['funder'] == 'World Vision':
        return 'world_vision'
    else:
        return 'other'

    
def categorize_installer(train):
    '''This function will go through
    every row in the installer column
    and if the value is equal to any of
    the top 7, will return those values.
    If not, will return other.'''
    if train['installer'] == 'DWE':
        return 'dwe'
    elif train['installer'] == 'Government':
        return 'govt'
    elif train['installer'] == 'RWE':
        return 'rwe'
    elif train['installer'] == 'Commu':
        return 'commu'
    elif train['installer'] == 'DANIDA':
        return 'danida'
    elif train['installer'] == 'KKKT':
        return 'kkkt'
    elif train['installer'] == 'Hesawa':
        return 'hesawa'
    else:
        return 'other'

    
def numeric_public(row):
    if row['public_meeting'] == True:
        return 1
    else:
        return 0

    
def categorize_scheme(row):
    '''This function will go through each
    row in the scheme management column
    and if the value is equal to any of the 
    top 7, will return those values. If not,
    will categorize the value as other.'''
    if row['scheme_management'] == 'VWC':
        return 'vwc'
    elif row['scheme_management'] == 'WUG':
        return 'wug'
    elif row['scheme_management'] == 'Water authority':
        return 'water_authority'
    elif row['scheme_management'] == 'WUA':
        return 'wua'
    elif row['scheme_management'] == 'Water Board':
        return 'water_board'
    elif row['scheme_management'] == 'Parastatal':
        return 'parastatal'
    elif row['scheme_management'] == 'Private operator':
        return 'private_operator'
    else:
        return 'other'

    
def permit(row):
    
    if row['permit'] == True:
        return 1
    else:
        return 0
    
def encode_lga(df):
    '''
    encodes the 'lga' column into the values
    'rural', 'urban', and 'other'
    
    returns DataFrame with column 'lga_coded'
    '''
    
    lga_e = []
    
    for entry in df.lga:
        key = entry.split()[-1]
        if key == 'Rural':
            lga_e.append('rural')
        elif key == 'Urban':
            lga_e.append('urban')
        else:
            lga_e.append('other')
    
    df['lga_coded'] = lga_e
    return df.drop('lga', axis=1)    
    

    
def categorize_contruction(row):
    if row['construction_year'] < 1970:
        return '1960s'
    elif row['construction_year'] < 1980:
        return '1970s'
    elif row['construction_year'] < 1990:
        return '1980s'
    elif row['construction_year'] < 2000:
        return '1990s'
    elif row['construction_year'] < 2010:
        return '2000s'
    elif row['construction_year'] < 2020:
        return '2010s'

    
def load_processed_train_df():
    '''This function takes the inital 
    dataframe created with the first 
    function in this file, imputes the 
    missing values in the dataframe, bins
    a few columns into categories, and
    removes columns that provide the same 
    information as another column in the
    dataframe'''
    train = load_train_add_target_df()
    #Creating the status column for numerically transformed status_group
    train['status'] = train.status_group.replace(numeric_status_group)
    #Binning funder field into 7 categories
    train['funder'] = train.apply(lambda x: categorize_funder(x), axis=1)
    #Binning installer field into 7 categories
    train['installer'] = train.apply(lambda x: categorize_installer(x), axis=1)
    #Removing subvillage field
    train = train.drop(columns=['subvillage'], axis=1)
    #Filling 9 percent of na values with False
    train['public_meeting'].fillna(False, limit=300, inplace=True)
    #Filling 91 percent of na values with True
    train['public_meeting'].fillna(True, inplace=True)
    #Binning scheme management field into 7 categories
    train['scheme_management'] = train.apply(lambda x: categorize_scheme(x), axis=1)
    #Removing scheme name field
    train = train.drop(columns=['scheme_name'], axis=1)
    #Filling 31 percent of na values with False
    train['permit'].fillna(False, limit=947, inplace=True)
    #Filling 69 percent of na values with True
    train['permit'].fillna(True, inplace=True)
    #Removing wpt name field
    train = train.drop(columns=['wpt_name'], axis=1)
    #Encode and bin region field and drop original region column
    train = encode_region(train)
    #Encode lga field
    train = encode_lga(train)
    #Removing recorded by field
    train = train.drop(columns=['recorded_by'], axis=1)
    #Changing permit field to numerical data
    train['permit'] = train.apply(lambda x: permit(x), axis=1)
    #Binning construction year by decade
    train['construction_year'] = train.apply(lambda x: categorize_contruction(x), axis=1)
    #Removing extraction type and extraction type group fields
    train = train.drop(columns=['extraction_type', 'extraction_type_group'], axis=1)
    #Removing management group field
    train = train.drop(columns=['management_group'], axis=1)
    #Removing payment type group field
    train = train.drop(columns=['payment_type'], axis=1)
    #Removing payment type field
    
    
   # ohe_features = ['funder', 'installer', 'basin', 
               #'region_code', 'district_code', 'lga', 'public_meeting',
               #'scheme_management', 'permit', 'construction_year', 
               #'extraction_type_class', 'management',
               #'payment', 'quality_group',
              # 'quantity', 'source', 'source_class', 'waterpoint_type']
    #cont_features = ['amount_tsh', 'gps_height', 
                 #'num_private', 'public_meeting',
                  #'population']
    
    #ohe = OneHotEncoder()
    #ss = StandardScaler()
    
    #train_cat = train[ohe_features]
    #train_cont = train[cont_features].astype(float)
    
    #train_ohe = ohe.fit_transform(train_cat)
    #train_scl = pd.DataFrame(ss.fit_transform(train_cont), columns=train[cont_features].columns)
    #columns = ohe.get_feature_names(input_features=train_cat.columns)
    #train_processed = pd.DataFrame(train_ohe.todense(), columns=columns)
    #train_all = pd.concat([train_scl, train_processed], axis=1)
    #train_all['status'] = train['status']
    
    return train

    
    
    
    
    




