import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


mat_df=pd.read_csv('student_mat.csv')
por_df=pd.read_csv('student_por.csv')


def one_hot_encoder(df):
    # Kategorik sütunları seçelim
    kat_sutunlar = df.select_dtypes(include="object").columns

    # Encoder
    encoder = OneHotEncoder(sparse_output=False)

    # Encode işlemi ve yeni DataFrame
    encoded_df = pd.DataFrame(encoder.fit_transform(df[kat_sutunlar]),
                              columns=encoder.get_feature_names_out(kat_sutunlar))

    encoded_df.head()
    yeni_df=df.drop(columns=kat_sutunlar)
    # Eski kategorik sütunları atıp encode edilmiş sütunları ekleyelim
    yeni_df = yeni_df.join(encoded_df)

    return yeni_df

onehot_mat_df=one_hot_encoder(mat_df)
onehot_por_df=one_hot_encoder(por_df)


def sicaklik_haritasi(df,df_isim):
    corr = df.corr()
    g1 = corr[abs(corr['G1']) >= 0.1]['G1'].drop('G1').sort_values(ascending=False)
    g2 = corr[abs(corr['G2']) >= 0.1]['G2'].drop('G2').sort_values(ascending=False)
    g3 = corr[abs(corr['G3']) >= 0.1]['G3'].drop('G3').sort_values(ascending=False)
    print(g1, g2, g3)
    list_x=[g1,g2,g3]
    i=1
    for x in list_x:
        plt.figure(figsize=(20, 18))
        sns.heatmap(x.to_frame(), cmap='coolwarm', annot=True)
        plt.title(f'{df_isim} G{i} için sıcaklık haritası')
        plt.savefig(f'{df_isim}{i}.png')
        plt.show()
        i+=1


sicaklik_haritasi(onehot_mat_df,'Matematik Verisi ')
sicaklik_haritasi(onehot_por_df,'Portekiz Verisi ')
