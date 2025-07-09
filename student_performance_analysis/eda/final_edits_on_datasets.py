import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


mat_df=pd.read_csv('student_mat.csv')
por_df=pd.read_csv('student_por.csv')




mat_selected_features = [
    "failures", "Medu", "Fedu", "higher", "romantic", "Mjob", "Fjob",
    "address", "sex", "goout", "traveltime", "age", "studytime",
    "Walc", "schoolsup", "internet", "paid","Basari"
]
len(mat_selected_features)

mat_df['G_ort']=(mat_df['G1']+mat_df['G2']+mat_df['G3'])/3
mat_df['Basari']=[1 if row>=10 else 0for row in mat_df['G_ort']]
binary33_target_mat_df=mat_df.drop(['G1','G2','G3','G_ort'], axis=1)


binary18_target_mat_df=mat_df[mat_selected_features]



por_selected_features = [
    "failures", "school","Medu", "Fedu", "higher", "Mjob", "Fjob",
    "address", "sex", "traveltime", "age", "studytime","reason","guardian",
    "Walc","Dalc" ,"internet", "absences","freetime","Basari"
]
len(por_selected_features)

por_df['G_ort']=(por_df['G1']+por_df['G2']+por_df['G3'])/3
por_df['Basari']=[1 if row>=10 else 0for row in por_df['G_ort']]
binary33_target_por_df=por_df.drop(['G1','G2','G3','G_ort'], axis=1)

binary20_target_por_df=por_df[por_selected_features]






def olceklendirme(df):

    # Sayısal sütunları seçiyoruz
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

scaled18_binary_target_mat_df=olceklendirme(binary18_target_mat_df)
scaled33_binary_target_mat_df=olceklendirme(binary33_target_mat_df)

scaled20_binary_target_por_df=olceklendirme(binary20_target_por_df)
scaled33_binary_target_por_df=olceklendirme(binary33_target_por_df)



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

ohe_scaled18_binary_target_mat_df=one_hot_encoder(scaled18_binary_target_mat_df)
ohe_scaled33_binary_target_mat_df=one_hot_encoder(scaled33_binary_target_mat_df)

ohe_scaled20_binary_target_por_df=one_hot_encoder(scaled20_binary_target_por_df)
ohe_scaled33_binary_target_por_df=one_hot_encoder(scaled33_binary_target_por_df)


ohe_scaled18_binary_target_mat_df.to_csv('ohe_scaled18_binary_target_mat_df.csv',index=False)
ohe_scaled33_binary_target_mat_df.to_csv('ohe_scaled33_binary_target_mat_df.csv',index=False)
ohe_scaled20_binary_target_por_df.to_csv('ohe_scaled20_binary_target_por_df.csv',index=False)
ohe_scaled33_binary_target_por_df.to_csv('ohe_scaled33_binary_target_por_df.csv',index=False)

