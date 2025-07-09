import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from altair import to_csv
from anyio.abc import value
from sqlalchemy.sql.functions import random
from sympy import print_tree
from tabulate import tabulate
import warnings
import os
import textwrap

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 4000)

mat_df=pd.read_csv('datasets/student-mat.csv')
por_df=pd.read_csv('datasets/student-por.csv')


def df_duzenleme(df,df_isim):

    """
    Bu fonksiyonun amacı yanlış şekilde oluşturulmuş veri setini işleyebileceğimiz şekilde düzenlemektir , bu verisetlerine özel olarak oluşturulmuştur
    """
    columns = df.columns[0]

    col_list = columns.split(';')

    print(f'{df_isim} Verisinin düzenlenmemiş halinden karışık 5 satır ')
    print(df.sample(n=5))

    rows = df[columns]
    all_data = []

    for row in rows:
        values = row.strip(';').split(';')  # remove trailing ';' if any
        all_data.append(values)

    df = pd.DataFrame(all_data, columns=col_list)
    print(f'{df_isim} Verisinin düzenlenmiş halinden karışık 5 satır ')
    print(df.sample(n=5))

    return df


mat_df=df_duzenleme(mat_df,'Matematik Verisi')
por_df=df_duzenleme(por_df,'Portekizce Verisi')

to_csv(mat_df,filename='duzenlenmis_mat_df.csv')
to_csv(por_df,filename='duzenlenmis_por_df.csv')






















