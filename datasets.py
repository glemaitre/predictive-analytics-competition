import pandas as pd


def create_dataframe_pac_2018(path_data_nii, path_data_xls):
    # read the Excel files
    df = pd.read_excel(path_data_xls)
    # add the path to the data into the dataframe
    df['t1'] = path_data_nii + df['PAC_ID'] + '.nii'
    # set the index
    df = df.set_index('PAC_ID')
    return df
