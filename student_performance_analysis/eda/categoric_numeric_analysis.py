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

mat_df=pd.read_csv('student_mat.csv')
por_df=pd.read_csv('student_por.csv')


def kat_num_analiz(df,df_isim,yazdir=True):
    # Kategorik sütunları belirleme
    kat_sutunlar = [col for col in df.columns if df[col].dtypes in ["category", "object"]]

    # Sayısal sütunları belirleme
    say_sutunlar = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]

    if yazdir:
        print("\n" + "=" * 150)
        print(f"📌 {df_isim} VERİSİ SÜTUNLAR İÇİN TÜR ANALİZİ\n".center(150))
        print("=" * 150 + "\n")

        # Genel bilgiler
        print(f"🔹 Toplam Sütun Sayısı: {df.shape[1]}")
        print(f"🔹 Kategorik Sütun Sayısı: {len(kat_sutunlar)}")
        print(f"🔹 Sayısal Sütun Sayısı: {len(say_sutunlar)}")

        # Unique değer sayılarının analizi
        print("📊 Kategorik Sütunların Unique Değer Sayıları:")
        print(df[kat_sutunlar].nunique(), "\n")

        print("📊 Sayısal Sütunların Unique Değer Sayıları:")
        print(df[say_sutunlar].nunique(), "\n")


    return kat_sutunlar, say_sutunlar



mat_kat,mat_num=kat_num_analiz(mat_df,'Matematik Verisi')
por_kat,por_num=kat_num_analiz(por_df,'Portekizce Verisi')
def kat_bool_ozet(df, df_isim, kat_sutunlar, sinif_limiti=10):
    print(f"\n{'=' * 200}")
    print(f"📊 {df_isim.upper()} VERİ SETİ KATEGORİK ANALİZİ".center(200))
    print(f"{'=' * 200}\n")
    for kat_sutun in kat_sutunlar:
        benzersiz_sayilar = df[kat_sutun].nunique()
        print(f"\n{'-' * 120}")
        print(f"{kat_sutun.upper()} DEĞİŞKENİ ANALİZİ (Unique Değerler: {benzersiz_sayilar})".center(120))
        print(f"{'-' * 120}\n")

        value_counts = df[kat_sutun].value_counts(normalize=True, dropna=False) * 100
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pasta Grafiği
        value_counts.plot.pie(
            autopct="%.1f%%", startangle=90, cmap="coolwarm", shadow=True,
            explode=[0.05] * len(value_counts), ax=axes[0]
        )
        axes[0].set_ylabel("")
        axes[0].set_title(f"{kat_sutun} - Pasta Grafiği", fontsize=16, fontweight="bold")

        # Histogram
        sns.histplot(df[kat_sutun], bins=min(20, benzersiz_sayilar), kde=False, ax=axes[1],
                     color="royalblue")
        axes[1].set_xlabel(kat_sutun, fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Frekans", fontsize=14)
        axes[1].set_title(f"{kat_sutun} - Histogram", fontsize=16, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=30, labelsize=10)

        plt.tight_layout()
        plt.show()

def num_ozet(df, df_isim, say_sutunlar):
    print(f"\n{'=' * 200}")
    print(f"📊 {df_isim.upper()} VERİ SETİ SAYISAL ANALİZİ".center(200))
    print(f"{'=' * 200}\n")

    if not say_sutunlar:
        print(f"⚠ {df_isim} veri setinde sayısal sütun bulunmamaktadır.")
        return

    for say_sutun in say_sutunlar:
        print(f"\n{'-' * 120}")
        print(f"{say_sutun.upper()} DEĞİŞKENİ ANALİZİ".center(120))
        print(f"{'-' * 120}\n")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        sns.histplot(df[say_sutun], bins=20, kde=True, ax=axes[0], color="royalblue")
        axes[0].set_xlabel(say_sutun, fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Frekans", fontsize=14)
        axes[0].set_title(f"{say_sutun} - Histogram", fontsize=16, fontweight="bold")

        # Boxplot
        sns.boxplot(x=df[say_sutun], ax=axes[1], color="salmon")
        axes[1].set_xlabel(say_sutun, fontsize=14, fontweight="bold")
        axes[1].set_title(f"{say_sutun} - Boxplot", fontsize=16, fontweight="bold")

        plt.tight_layout()
        plt.show()



kat_bool_ozet(mat_df,'Matematik Verisinin Kategorik Sütunların Görselleştirilmesi',mat_kat)
num_ozet(mat_df,'Matematik Verisinin Sayısal Sütunların Görselleştirilmesi',mat_num)

kat_bool_ozet(por_df,'Portekiz Verisinin Kategorik Sütunların Görselleştirilmesi',por_kat)
num_ozet(por_df,'Portekiz Verisinin Sayısal Sütunların Görselleştirilmesi',por_num)

