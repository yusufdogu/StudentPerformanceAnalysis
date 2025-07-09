import pandas as pd
from altair import to_csv
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.serialization import MAP_SHARED

from student_performance_dataset import mat_df, por_df

mat_df=pd.read_csv('g1_g2_sayisal_duzenli_mat_df')
por_df=pd.read_csv('tek_target_g1_g2_sayisal_por_df.csv')

iki_deger=[]
coklu_deger=[]
for column in por_df.columns:
    if por_df[column].dtype == 'object':
        if por_df[column].nunique()==2:
            iki_deger.append(column)
        elif por_df[column].nunique()<=5 and por_df[column].nunique()>2:
            coklu_deger.append(column)


print(iki_deger,coklu_deger)

def iki_unique_p_value(df,column):
    por_df[column] = mat_df[column].apply(lambda x: str(x).replace('"', '').replace("'", "").strip())

    values=df[column].unique()
    print(values)
    """g1_v1 = por_df[por_df[column] == values[0] ]['G1']
    g1_v2 = por_df[por_df[column] == values[1] ]['G1']

    g2_v1 = por_df[por_df[column] == values[0] ]['G2']
    g2_v2 = por_df[por_df[column] == values[1] ]['G2']

    g3_v1 = por_df[por_df[column] == values[0] ]['G3']
    g3_v2 = por_df[por_df[column] == values[1] ]['G3']"""
    g1_v1 = por_df[por_df[column] == values[0]]['Basari']
    g1_v2 = por_df[por_df[column] == values[1]]['Basari']

    # T-test
    t_stat_1, p_val_1 = ttest_ind(g1_v1, g1_v2)
    """t_stat_2, p_val_2 = ttest_ind(g2_v1, g2_v2)
    t_stat_3, p_val_3 = ttest_ind(g3_v1, g3_v2)"""

    print(f"G1 comparison: t-statistic = {t_stat_1}, p-value = {p_val_1}")
    """print(f"G2 comparison: t-statistic = {t_stat_2}, p-value = {p_val_2}")
    print(f"G3 comparison: t-statistic = {t_stat_3}, p-value = {p_val_3}")"""


for col in iki_deger:
    print(f"{col} için istatistikler")
    iki_unique_p_value(por_df,col)
    print("")


from scipy.stats import f_oneway

def dinamik_anova(df, column):
    # Temizleme (eğer string karakter içeriyorsa)
    df[column] = df[column].apply(lambda x: str(x).replace('"', '').replace("'", "").strip())

    # Kaç farklı kategori olduğunu bul
    unique_vals = df[column].unique()
    print(f"Gruplar: {unique_vals}")

    # Her bir target için grupları hazırla
    g1_groups = [df[df[column] == val]['Basari'] for val in unique_vals]
    """g2_groups = [df[df[column] == val]['G2'] for val in unique_vals]
    g3_groups = [df[df[column] == val]['G3'] for val in unique_vals]

    # ANOVA testleri
    
    f_stat_2, p_val_2 = f_oneway(*g2_groups)
    f_stat_3, p_val_3 = f_oneway(*g3_groups)"""
    f_stat_1, p_val_1 = f_oneway(*g1_groups)

    # Sonuçlar
    print(f"G1 ANOVA: F-statistic = {f_stat_1:.4f}, p-value = {p_val_1:.4f}")
    """print(f"G2 ANOVA: F-statistic = {f_stat_2:.4f}, p-value = {p_val_2:.4f}")
    print(f"G3 ANOVA: F-statistic = {f_stat_3:.4f}, p-value = {p_val_3:.4f}")"""


for col in coklu_deger:
    print(f"{col} için istatistikler")
    dinamik_anova(por_df,col)
    print("")






corr = mat_df.corr()
g1 = corr[abs(corr['G1'])>=0.1]['G1'].drop('G1').sort_values(ascending=False)