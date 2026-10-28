import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. VERİ HAZIRLAMA VE TEMİZLEME

# Veri setini yükleme
df = pd.read_csv("housing_price_dataset.csv")

# Kullanılmayacak sütunları veri setinden çıkarma
df = df.drop(columns=['Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt'], errors='ignore')

# Price ve SquareFeet sütunlarını güvenli şekilde sayısala dönüştürme
# (Eğer veri metin olarak geldiyse sadece ilk 6 basamağını alarak mantıklı fiyat aralığına çekiyoruz)
def fiyati_duzelt(deger):
    s = str(deger).replace('.', '').replace(',', '').strip()
    if len(s) > 6:
        s = s[:6]
    return float(s) if s.isdigit() else np.nan

df['Price'] = df['Price'].apply(fiyati_duzelt)
df['SquareFeet'] = pd.to_numeric(df['SquareFeet'], errors='coerce')

# Metrekareye dönüştürme
df['Metrekare'] = df['SquareFeet'] * 0.092903

# Mantıksız değerleri veya boşlukları temizleme
df = df.dropna(subset=['Price', 'Metrekare'])
df = df[(df['Price'] > 10000) & (df['Price'] < 5000000)] 

# Dönüştürme sonrası eksik kalan değerleri medyan ile doldurma
df['Metrekare'] = df['Metrekare'].fillna(df['Metrekare'].median())
df['Price'] = df['Price'].fillna(df['Price'].median())

# 2. AYKIRI DEĞERLERİ TEMİZLEME (IQR YÖNTEMİ)

def iqr_baskila(dataframe, sutun):
    """Sütundaki aykırı değerleri tespit edip alt ve üst sınırlara baskılar."""
    Q1 = dataframe[sutun].quantile(0.25)
    Q3 = dataframe[sutun].quantile(0.75)
    IQR = Q3 - Q1
    
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    
    # Alt/üst sınır dışındaki verileri sınırlara eşitleme
    dataframe[sutun] = np.where(dataframe[sutun] < alt_sinir, alt_sinir, dataframe[sutun])
    dataframe[sutun] = np.where(dataframe[sutun] > ust_sinir, ust_sinir, dataframe[sutun])
    
    return dataframe

# Aykırı değerleri temizleme
df = iqr_baskila(df, 'Metrekare')
df = iqr_baskila(df, 'Price')

# Metrekare dağılımı için Boxplot grafiği oluşturup kaydetme
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Metrekare'])
plt.title('Metrekare Boxplot Grafiği (Aykırı Değerler Temizlenmiş)')
plt.savefig("metrekare_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()


# 3. MODEL EĞİTİMİ VE DEĞERLENDİRME


# X (Girdi) ve y (Hedef) değişkenlerini belirleme
X = df[['Metrekare']]  
y = df['Price']

# Veriyi %80 Eğitim, %20 Test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doğrusal Regresyon modelini kurma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinden başarı metriklerini hesaplama
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model R2 Skoru: {model.score(X_test, y_test):.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.2f}")


# 4. GÖRSELLEŞTİRME 


plt.figure(figsize=(10, 6))

# Gerçek veri noktaları
sns.scatterplot(x=df['Metrekare'], y=df['Price'], alpha=0.4, color='blue', label='Gerçek Veriler')

# Modelin regresyon tahmin çizgisi
x_cizgi = np.linspace(df['Metrekare'].min(), df['Metrekare'].max(), 100).reshape(-1, 1)
y_cizgi = model.predict(pd.DataFrame(x_cizgi, columns=['Metrekare']))

plt.plot(x_cizgi, y_cizgi, color='red', linewidth=2, label='Linear Regression Tahmin Çizgisi')

plt.title('Ev Büyüklüğü ve Fiyat İlişkisi (Regresyon Çizgisi İle)')
plt.xlabel('Ev Büyüklüğü (Metrekare - m²)')
plt.ylabel('Fiyat (Price)')
plt.legend()
plt.grid(True)

# Grafiği dosyaya kaydetme
plt.savefig("regresyon_scatter_birlesik.png", dpi=300, bbox_inches='tight')
plt.close()

print("Grafikler başarıyla kaydedildi.")

# 5. KULLANICIDAN GİRDİ ALARAK TAHMİN YAPMA

ev_buyuklugu = float(input("\nEv büyüklüğünü giriniz (Metrekare - m²): "))

# Sütun ismi uyarısı almamak için girdiyi DataFrame formatında veriyoruz
girdi_df = pd.DataFrame([[ev_buyuklugu]], columns=['Metrekare'])
tahmin_edilen_fiyat = model.predict(girdi_df)

print(f"Tahmin Edilen Fiyat: {tahmin_edilen_fiyat[0]:.2f} TL")