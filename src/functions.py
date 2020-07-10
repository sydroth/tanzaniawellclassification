import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import textdistance
from collections import Counter
import os
import pandas as pd
import numpy as np
import pickle

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
    train['status'] = train.status_group.replace(numeric_status_group())

    train = combine_extraction(train)
    train = combine_managements(train)
    train = combine_installer_funder(train)
    train = combine_waterpoint(train)
    train = clean_permit(train)
    train = create_decades(train)
    train = fill_years(train)
    
    #Encode and bin region field and drop original region column
    train = encode_region(train)
    #Encode lga field
    train = encode_lga(train)


    #Filling public meeting
    train['public_meeting'].fillna('not_known', inplace=True)
    
    return train

    
    
    
"""
==================================================================================================================
Max's cleaning Functions
==================================================================================================================
"""

def average_prediction(construction_year):
    if type(construction_year) == str:
        decade_split = construction_year.split('-')
        return pd.Series(decade_split).map(int).mean()
    else:
        return construction_year


def combiner(row, col_1, col_2):
    if row[col_1]!=row[col_2]:
        return f'{row[col_1]}/{row[col_2]}'
    else:
        return row[col_1]
    
def fill_unknown(row, col_1, col_2, unknown):
    if (row[col_1] in unknown) &\
       (row[col_2] in unknown):
        row[col_1] = 'unknown'
        row[col_2] = 'unknown'
        return row
    elif row[col_1] in unknown:
        row[col_1] = row[col_2]
    elif row[col_2] in unknown:
        row[col_2] = row[col_1]
    return row


def combine_managements(df):
    col_1 = 'scheme_management'
    col_2 = 'management'

    df[col_1] = df[col_1].fillna('na')
    df[col_2] = df[col_2].fillna('na')

    df[col_2] = df[col_2].map(lambda x: x.lower())
    df[col_1] = df[col_1].map(lambda x: x.lower())

    df = df.apply(lambda row: fill_unknown(row, col_1, col_2, ['na', 'other', 'none', 'unknown']), axis=1)
    df['scheme_management/management'] = df.apply(lambda row: combiner(row, col_1, col_2), axis=1)
    top = df['scheme_management/management'].value_counts()[df['scheme_management/management'].value_counts()>100]
    
    df['scheme_management/management'] = df['scheme_management/management'].map(lambda x: x if x in top.index else 'binned')
    df.drop([col_1, col_2], axis=1, inplace=True)
    
    return df


def combine_waterpoint(df):
    df['waterpoint_type/group'] = df.apply(lambda row: combiner(row, 'waterpoint_type', 'waterpoint_type_group'), axis=1)

    df['waterpoint_type/group'].value_counts()
    df.drop(['waterpoint_type', 'waterpoint_type_group'], axis=1, inplace=True)
    return df


misspellings = {'dwe&': 'dwe',
 'dwe': 'dwe',
 'dwe/': 'dwe',
 'dwe}': 'dwe',
 'dw#': 'dwe',
 'dw$': 'dwe',
 'dw': 'dwe',
 'dw e': 'dwe',
 'dawe': 'dwe',
 'dweb': 'dwe',
 'government': 'central government',
 'government of tanzania': 'central government',
 'gove': 'central government',
 'tanzanian government': 'central government',
 'governme': 'central government',
 'goverm': 'central government',
 'tanzania government': 'central government',
 'cental government': 'central government',
 'gover': 'central government',
 'centra government': 'central government',
 'go': 'central government',
 'centr': 'central government',
 'central govt': 'central government',
 'cebtral government': 'central government',
 'governmen': 'central government',
 'govern': 'central government',
 'central government': 'central government',
 'olgilai village community': 'community',
 'maseka community': 'community',
 'kitiangare village community': 'community',
 'sekei village community': 'community',
 'igolola community': 'community',
 'comunity': 'community',
 'mtuwasa and community': 'community',
 'village community members': 'community',
 'district community j': 'community',
 'marumbo community': 'community',
 'ngiresi village community': 'community',
 'community': 'community',
 'village community': 'community',
 'commu': 'community',
 'ilwilo community': 'community',
 'communit': 'community',
 'taboma/community': 'community',
 'oldadai village community': 'community',
 'villagers': 'community',
 'kkkt': 'kkkt',
 'kkkt dme': 'kkkt',
 'kkkt-dioces ya pare': 'kkkt',
 'kkkt katiti juu': 'kkkt',
 'kkkt leguruki': 'kkkt',
 'kkkt mareu': 'kkkt',
 'kkkt ndrumangeni': 'kkkt',
 'kk': 'kkkt',
 'kkkt church': 'kkkt',
 'kkkt kilinga': 'kkkt',
 'kkkt canal': 'kkkt',
 'kkt': 'kkkt',
 'lutheran church': 'kkkt',
 'luthe': 'kkkt',
 'haidomu lutheran church': 'kkkt',
 'world vision': 'world vision',
 'world vission': 'world vision',
 'world visiin': 'world vision',
 'world division': 'world vision',
 'world': 'world vision',
 'world nk': 'world vision',
 'district council': 'district council',
 'district counci': 'district council',
 'district  council': 'district council',
 'mbozi district council': 'district council',
 'wb / district council': 'district council',
 'mbulu district council': 'district council',
 'serengeti district concil': 'district council',
 'district water department': 'district council',
 'tabora municipal council': 'district council',
 'hesawa': 'hesawa',
 'esawa': 'hesawa',
 'hesaw': 'hesawa',
 'unknown installer': 'unknown'}

def bin_installer(df):
    """
    input: dataframe
    output: returns a new dataframe with a new column, installer_binned, that has a cleaned installer row
    """
    new_df = df.copy()
    new_df['installer_binned'] = new_df['installer']
    new_df['installer_binned'] = new_df['installer_binned'].fillna('unknown')
    new_df['installer_binned'] = new_df['installer_binned'].map(lambda x: x.lower())
    new_df['installer_binned'] = new_df['installer_binned'].replace(misspellings)
    
    new_df.drop(['installer'], axis=1, inplace=True)
    
    return new_df

def bin_funder(df):
    """
    input: dataframe
    output: returns a new dataframe with a new column, installer_binned, that has a cleaned installer row
    """
    new_df = df.copy()
    new_df['funder_binned'] = new_df['funder']
    new_df['funder_binned'] = new_df['funder_binned'].fillna('unknown')
    new_df['funder_binned'] = new_df['funder_binned'].map(lambda x: x.lower())
    new_df['funder_binned'] = new_df['funder_binned'].replace(misspellings)
    
    new_df.drop(['funder'], axis=1, inplace=True)
    
    return new_df


def combine_installer_funder(df):
    new_df = bin_installer(df)
    new_df = bin_funder(new_df)

    col_1 = 'funder_binned'
    col_2 = 'installer_binned'

    new_df[col_1] = new_df[col_1].fillna('na')
    new_df[col_2] = new_df[col_2].fillna('na')

    new_df[col_2] = new_df[col_2].map(lambda x: x.lower())
    new_df[col_1] = new_df[col_1].map(lambda x: x.lower())


    new_df = new_df.apply(lambda row: fill_unknown(row, col_1, col_2, ['0', 'other', 'unknown']), axis=1)
    new_df['funder/installer'] = new_df.apply(lambda row: combiner(row, col_1, col_2), axis=1)


    top = new_df['funder/installer'].value_counts()[new_df['funder/installer'].value_counts() > 200]
    new_df['funder/installer'] = new_df['funder/installer'].map(lambda x: x if x in top.index else 'binned')
    new_df.drop([col_1, col_2], axis=1, inplace=True)
    
    return new_df

def clean_permit(df):
    df['permit'].fillna('not_known')
    df['permit'] = df['permit'].map(str)
    
    return df



def combine_extraction(df):
    col_1 = 'extraction_type'
    col_2 = 'extraction_type_group'

    df[col_1] = df[col_1].fillna('na')
    df[col_2] = df[col_2].fillna('na')

    df[col_2] = df[col_2].map(lambda x: x.lower())

    df[col_1] = df[col_1].map(lambda x: x.lower())


    df['extraction_type/group'] = df.apply(lambda row: combiner(row, col_1, col_2), axis=1)


    top = df['extraction_type/group'].value_counts()[:100]
    top

    df['extraction_type/group'] = df['extraction_type/group'].map(lambda x: x if x in top.index else 'binned')
    df.drop([col_1, col_2], axis=1, inplace=True)
    
    # Extraction iteration two

    col_1 = 'extraction_type_class'
    col_2 = 'extraction_type/group'

    df[col_1] = df[col_1].fillna('na')
    df[col_2] = df[col_2].fillna('na')

    df[col_2] = df[col_2].map(lambda x: x.lower())

    df[col_1] = df[col_1].map(lambda x: x.lower())

    df[df[col_2]==df[col_1]]

    df['extraction_type/group/class'] = df.apply(lambda row: combiner(row, col_1, col_2), axis=1)
    df.drop([col_1, col_2], axis=1, inplace=True)
    
    return df
    
def bin_year(year):
    if year<1960:
        return 'unknown'
    elif year>=1960 and year<1970:
        return '1960-1970'
    elif year>=1970 and year<1980:
        return '1970-1980'
    elif year>=1980 and year<1990:
        return '1980-1990'
    elif year>=1990 and year<2000:
        return '1990-2000'
    elif year>=2000 and year<2010:
        return '2000-2010'
    elif year>=2010 and year<2020:
        return '2010-2020'
    else:
        return year
    
def create_decade_columns(decade, target_decade):
    if decade == target_decade:
        return 1
    else:
        return 0
    
    
    
def encode_and_concat_feature_train(X_train_all_features, feature_name):
    """
    Helper function for transforming training data.  It takes in the full X dataframe and
    feature name, makes a one-hot encoder, and returns the encoder as well as the dataframe
    with that feature transformed into multiple columns of 1s and 0s
    """
    # make a one-hot encoder and fit it to the training data
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
    single_feature_df = X_train_all_features[[feature_name]]
    ohe.fit(single_feature_df)
    
    # call helper function that actually encodes the feature and concats it
    X_train_all_features = encode_and_concat_feature(X_train_all_features, feature_name, ohe)
    
    return ohe, X_train_all_features

def encode_and_concat_feature(X, feature_name, ohe):
    """
    Helper function for transforming a feature into multiple columns of 1s and 0s. Used
    in both training and testing steps.  Takes in the full X dataframe, feature name, 
    and encoder, and returns the dataframe with that feature transformed into multiple
    columns of 1s and 0s
    """
    # create new one-hot encoded df based on the feature
    single_feature_df = X[[feature_name]]
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0])
    
    # drop the old feature from X and concat the new one-hot encoded df
    X = X.drop(feature_name, axis=1)
    X = pd.concat([X, ohe_df], axis=1)
    
    return X


def create_decades(df):
    df['decade'] = df['construction_year'].map(bin_year)

    non_zeros = df[df['construction_year']>0]
    decades = ['2000-2010',
    '1990-2000',
    '1980-1990',
    '2010-2020',
    '1970-1980',
    '1960-1970']
    for decade in decades:
        non_zeros[decade] = non_zeros['decade'].map(lambda val: create_decade_columns(val, decade))
    
    non_zeros = non_zeros.loc[:,decades]
    
    
    
    zeros = df[df['construction_year']==0]
    zeros_index = zeros.index
    zeros = zeros.reset_index().drop("index", axis=1)

#     zeros_clean = combine_managements(zeros)
#     zeros_clean = combine_waterpoint(zeros_clean)
#     zeros_clean = clean_permit(zeros_clean)
#     zeros_clean = combine_installer_funder(zeros_clean)
#     zeros_clean = combine_extraction(zeros_clean)

    zeros_clean = zeros.loc[:,['waterpoint_type/group', 'scheme_management/management', 'basin', 'region_code', 
                        'funder/installer', 'extraction_type/group/class',  'source', 'source_type', 
                       'management_group', 'permit', 'district_code']]

    funder_installer_ohe = pickle.load(open('../src/funder_installer_ohe.sav', 'rb'))
    extraction_type_group_class_ohe = pickle.load(open('../src/extraction_type_group_class_ohe.sav', 'rb'))
    scheme_management_management_ohe = pickle.load(open('../src/scheme_management_management_ohe.sav', 'rb'))
    management_group_ohe = pickle.load(open('../src/management_group_ohe.sav', 'rb'))
    source_type_ohe = pickle.load(open('../src/source_type_ohe.sav', 'rb'))
    source_ohe = pickle.load(open('../src/source_ohe.sav', 'rb'))
    waterpoint_type_group_ohe = pickle.load(open('../src/waterpoint_type_group_ohe.sav', 'rb'))
    permit_ohe = pickle.load(open('../src/permit_ohe.sav', 'rb'))
    basin_ohe = pickle.load(open('../src/basin_ohe.sav', 'rb'))
    district_code_ohe = pickle.load(open('../src/district_code_ohe.sav', 'rb'))
    region_code_ohe = pickle.load(open('../src/region_code_ohe.sav', 'rb'))

    X_test_all_features = zeros_clean.copy().reset_index().drop("index", axis=1)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, 'funder/installer', funder_installer_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "extraction_type/group/class", extraction_type_group_class_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "scheme_management/management", scheme_management_management_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "management_group", management_group_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "source_type", source_type_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "source", source_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "waterpoint_type/group", waterpoint_type_group_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "permit", permit_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "basin", basin_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "district_code", district_code_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "region_code", region_code_ohe)


    model = pickle.load(open('../src/year_predictor.sav', 'rb'))

    predictions_ = model.predict_proba(X_test_all_features)
    prediction_df = pd.DataFrame(predictions_, columns=model.classes_, index=zeros_index)
    

    full_df = pd.concat([non_zeros, prediction_df])
    full_df.sort_index(inplace=True)
    
    new_df = df.merge(full_df, left_index=True, right_index=True)
    new_df.drop('decade', axis=1, inplace=True)
    return new_df



def fill_years(df):
    df['decade'] = df['construction_year'].map(bin_year)

    non_zeros = df[df['construction_year']>0]
    decades = ['2000-2010',
    '1990-2000',
    '1980-1990',
    '2010-2020',
    '1970-1980',
    '1960-1970']
    for decade in decades:
        non_zeros[decade] = non_zeros['decade'].map(lambda val: create_decade_columns(val, decade))

    non_zeros = non_zeros.loc[:,['construction_year']]

    zeros = df[df['construction_year']==0]
    zeros_index = zeros.index
    zeros = zeros.reset_index().drop("index", axis=1)


    zeros_clean = zeros.loc[:,['waterpoint_type/group', 'scheme_management/management', 'basin', 'region_code', 
                        'funder/installer', 'extraction_type/group/class',  'source', 'source_type', 
                       'management_group', 'permit', 'district_code']]

    funder_installer_ohe = pickle.load(open('../src/funder_installer_ohe.sav', 'rb'))
    extraction_type_group_class_ohe = pickle.load(open('../src/extraction_type_group_class_ohe.sav', 'rb'))
    scheme_management_management_ohe = pickle.load(open('../src/scheme_management_management_ohe.sav', 'rb'))
    management_group_ohe = pickle.load(open('../src/management_group_ohe.sav', 'rb'))
    source_type_ohe = pickle.load(open('../src/source_type_ohe.sav', 'rb'))
    source_ohe = pickle.load(open('../src/source_ohe.sav', 'rb'))
    waterpoint_type_group_ohe = pickle.load(open('../src/waterpoint_type_group_ohe.sav', 'rb'))
    permit_ohe = pickle.load(open('../src/permit_ohe.sav', 'rb'))
    basin_ohe = pickle.load(open('../src/basin_ohe.sav', 'rb'))
    district_code_ohe = pickle.load(open('../src/district_code_ohe.sav', 'rb'))
    region_code_ohe = pickle.load(open('../src/region_code_ohe.sav', 'rb'))

    X_test_all_features = zeros_clean.copy().reset_index().drop("index", axis=1)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, 'funder/installer', funder_installer_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "extraction_type/group/class", extraction_type_group_class_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "scheme_management/management", scheme_management_management_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "management_group", management_group_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "source_type", source_type_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "source", source_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "waterpoint_type/group", waterpoint_type_group_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "permit", permit_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "basin", basin_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "district_code", district_code_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "region_code", region_code_ohe)

    model = pickle.load(open('../src/year_predictor.sav', 'rb'))

    predictions_ = model.predict(X_test_all_features)
    prediction_df = pd.DataFrame(predictions_, index=zeros_index, columns=['construction_year'])

    full_df = pd.concat([non_zeros, prediction_df])
    full_df.sort_index(inplace=True)

    full_df['construction_year'] = full_df['construction_year'].map(average_prediction)

    new_df = df.drop('construction_year', axis=1).merge(full_df, left_index=True, right_index=True)
    new_df.drop('decade', axis=1, inplace=True)

    return new_df