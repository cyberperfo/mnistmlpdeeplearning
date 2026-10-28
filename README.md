# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu depo, PyTorch kütüphanesi kullanılarak MNIST veri seti üzerinde Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisiyle el yazısı rakam sınıflandırması gerçekleştirmek ve modelin gerçek dünya verileri üzerindeki başarımını incelemek amacıyla geliştirilmiştir.

---

## 1. Proje Amacı ve Kapsamı
Ödevin temel amacı;
* PyTorch ile derin öğrenme altyapısını kurabilmek,
* Temel bir **MLP (Multi-Layer Perceptron)** modeli tasarlamak (CNN kesinlikle kullanılmamıştır),
* Veri setini Train, Validation ve Test olarak doğru bir şekilde ayırmak,
* Model hiperparametrelerinin (Epoch, Batch Size, Learning Rate vb.) performans üzerindeki etkilerini gözlemlemek,
* Öğrenci tarafından kendi el yazısıyla oluşturulan gerçek dünya verileri (okul numarası) üzerinde modelin genelleme yeteneğini (`distribution shift`) test etmektir.

---

## 2. Proje Mimarisi ve Hiperparametreler
Model sıfırdan tasarlanmış olup şu katman yapısına ve hiperparametre değerlerine sahiptir:

* **Model Katmanları:**
  * **Giriş Katmanı (Flatten):** $28 \times 28 = 784$ piksel düzleştirilmiş vektör.
  * **1. Gizli Katman (Hidden Layer 1):** `nn.Linear(784, 256)` + `nn.ReLU()`
  * **2. Gizli Katman (Hidden Layer 2):** `nn.Linear(256, 128)` + `nn.ReLU()`
  * **Çıkış Katmanı (Output Layer):** `nn.Linear(128, 10)` (0-9 sınıf olasılıkları)

* **Hiperparametreler:**
  * **Epoch Sayısı:** 10
  * **Batch Size:** 64
  * **Öğrenme Oranı (Learning Rate):** 0.001
  * **Optimizer:** Adam
  * **Loss Fonksiyonu:** CrossEntropyLoss

---

## 3. Veri Bölünmesi ve Eğitim Sonuçları
* MNIST eğitim verisinin **%80'i eğitim (train)**, **%20'si doğrulama (validation)** için ayrılmıştır. Test verisi model eğitim süreçlerinden tamamen bağımsız tutulmuştur.
* **Elde Edilen Başarım Metrikleri:**
  * **Final Training Loss / Accuracy:** ~0.0215 / %99.25
  * **Final Validation Loss / Accuracy:** ~0.0945 / %97.93
  * **Test Accuracy:** **%98.02**
  * **Toplam Eğitim Süresi:** ~112 saniye (CPU üzerinde)

---

## 4. Kendi El Yazısı Verisi ile Gerçek Dünya Testi (Distribution Shift)
Modelin gerçek hayattaki el yazısı verileri üzerindeki başarımını test etmek amacıyla kendi el yazımla yazdığım **24120205061** okul numarama ait rakamlar şu ön işleme adımlarından geçirilmiştir:
1. Görüntüler gri tonlamalı (`grayscale`) hale getirilmiştir.
2. MNIST veri dağılımına uyum sağlaması için renkler ters çevrilmiş (siyah arka plan, beyaz rakam) ve normalizasyon uygulanmıştır.
3. Her bir rakam tam olarak $28 \times 28$ piksel boyutuna getirilmiştir.

* **Değerlendirme:** Model, standart MNIST veri setinde yüksek bir doğruluk oranına (%98.02) ulaşmasına rağmen; bireysel el yazısındaki çizim kalınlığı, eğim ve biçim farkları gibi `distribution shift` (dağılım kayması) faktörleri nedeniyle gerçek dünya verilerinde ön işleme adımlarının ve veri çeşitliliğinin ne kadar kritik olduğu gözlemlenmiştir.

---

