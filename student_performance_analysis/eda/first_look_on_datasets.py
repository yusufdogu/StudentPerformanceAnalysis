import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)

mat_df=pd.read_csv('student_mat.csv')
por_df=pd.read_csv('student_por.csv')


def veriye_ilk_bakis(df,df_isim):
    """
    Bu fonksiyonun amacı veri setlerine ilk bakışı oluşturmaktır ; verisetinden örnekler , eksik değer bilgisi gibi bilgileri verir
    """
    separator = "=" * 100
    sub_separator = "-" * 80

    print(f"\n{separator}")
    print(f"🟢 ANALİZİ YAPILAN VERİ SETİ : {df_isim} 🟢".center(100))
    print(f"{separator}\n")

    #Veri boyutlarını alırız
    print("🔹 VERİ BOYUTLARI")
    print(f"   ➡ SATIR SAYISI: {df.shape[0]}")
    print(f"   ➡ SÜTUN SAYISI: {df.shape[1]}")
    print(sub_separator)

    #Sütun veri türlerini alırız
    print("\n🔹 DEĞİŞKEN (SÜTUN) TÜRLERİ")
    print(df.dtypes)
    print(sub_separator)

    #Veriden karışık 5 örnek gösterilir
    print("\n🔹 VERİDEN KARIŞIK ŞEKİLDE 5 ÖRNEK ".format(20))
    print(df.sample(20))
    print("\n")
    print(sub_separator)

    #Eksik değerleri sütun ve toplam veri seti bazında verir
    print("\n🔹 EKSİK DEĞERLER")
    eksik_degerler = df.isnull().sum()
    toplam_eksik_degerler = eksik_degerler.sum()

    if toplam_eksik_degerler == 0:
        print("✅ EKSİK DEĞER BULUNAMADI")
    else:
        print(eksik_degerler[eksik_degerler > 0])
    print(sub_separator)

    #Sütunların benzersiz değerlerini verir
    print("\n🔹 BENZERSİZ (UNIQUE) DEĞER SAYISI")
    print(df.nunique())
    print(sub_separator)

    #Sayısal sütunlar için istatistik verileri yansıtır diğer tür veriler içinse frekans ve benzersiz değer sayıları gibi değerleri verir
    print("\n🔹 VERİNİN İSTATİSTİK ÖZETİ")
    print(df.describe(percentiles=[0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print(f"{separator}\n")



veriye_ilk_bakis(mat_df,'Matematik Verisi')
veriye_ilk_bakis(por_df,'Portekizce Verisi')