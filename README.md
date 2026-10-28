# Ev Fiyatı Tahminleme Modeli (House Price Prediction) 🏡

Bu proje, evlerin metrekare (m²) büyüklüklerine dayanarak fiyatlarını tahmin eden bir Makine Öğrenmesi (Doğrusal Regresyon - Linear Regression) modelidir. 

## 📌 Proje Özeti
Kaggle üzerinden alınan ham veri seti üzerinde veri temizleme işlemleri yapılmış, aykırı değerler IQR yöntemi ile baskılanmış ve scikit-learn kullanılarak bir regresyon modeli eğitilmiştir. Kullanıcıdan alınan metrekare bilgisine göre tahmini ev fiyatı hesaplanmaktadır.

## 🚀 Kullanılan Teknolojiler
* **Python**
* **Pandas & NumPy:** Veri işleme, temizleme ve matematiksel hesaplamalar.
* **Scikit-Learn (sklearn):** Doğrusal Regresyon modeli eğitimi ve hata ölçümü (RMSE, R2).
* **Matplotlib & Seaborn:** Veri görselleştirme (Boxplot ve Scatter/Regresyon grafikleri).

## 🧠 Model Performansı ve Kısıtlamalar
* **Başarı Oranı (R² Skoru):** Modelin doğruluk (açıklanabilirlik) oranı yaklaşık **%15** civarındadır ($R^2 \approx 0.15$). 
* **Neden Düşük?** Gerçek dünyada gayrimenkul piyasasında bir evin fiyatını yalnızca metrekaresi belirlemez. Başlangıç seviyesi bir analiz yapmak adına veri setindeki `Bedrooms` (Oda Sayısı), `Bathrooms` (Banyo), `Neighborhood` (Mahalle/Konum) ve `YearBuilt` (Bina Yaşı) gibi çok önemli değişkenler modelden çıkarılmıştır. Örneğin; lüks bir konumdaki küçük bir ev, kötü bir konumdaki büyük bir evden daha pahalı olabileceği için sadece metrekareye dayalı tahminlerin sapma payı (RMSE) doğal olarak yüksektir.
* **Gelecekteki Geliştirmeler:** Modelin başarı oranını daha yüksek seviyelere çıkarmak için, çıkarılan bu diğer niteliklerin de dâhil edildiği **Çoklu Doğrusal Regresyon (Multiple Linear Regression)** uygulanarak proje ileri seviyeye taşınabilir.

## 📊 Veri Seti Hakkında
Boyut kısıtlamaları ve lisans nedeniyle veri seti bu depoya yüklenmemiştir. Projede kullanılan orijinal ham veri setine Kaggle üzerinden aşağıdaki linkten ulaşabilirsiniz:
* [https://www.kaggle.com/code/willianoliveiragibin/real-estate/input](https://www.kaggle.com/code/willianoliveiragibin/real-estate/input)

## 🛠️ Nasıl Çalıştırılır?
1. Kodları bilgisayarınıza indirin.
2. İlgili kütüphanelerin yüklü olduğundan emin olun (`pip install pandas numpy scikit-learn matplotlib seaborn`).
3. Kaggle'dan indirdiğiniz veri setini `housing_price_dataset.csv` adıyla `mo.py` ile aynı klasöre koyun.
4. Terminal veya komut satırından kodu çalıştırın: `python mo.py`
5. Ekrana evin metrekare (m²) cinsinden büyüklüğünü girerek tahmin sonucunu alabilirsiniz.

## 📈 Çıktılar ve Görselleştirme
Proje çalıştırıldığında;
* Verideki aykırı değerlerin temizlendiğini gösteren Boxplot grafiği oluşturulur.
* Ev büyüklüğü ve fiyat ilişkisini gösteren, üzerinde tahmin doğrusunun (regression line) bulunduğu bir Scatter plot kaydedilir.
