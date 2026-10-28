import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# ==========================================
# 1. CİHAZ AYARI (GPU / CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# ==========================================
# 2. VERİ SETİ HAZIRLIĞI VE ÖN İŞLEME
# ==========================================
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# MNIST veri setini indir (Bulunduğun dizindeki './data' klasörüne kaydeder)
full_train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Train ve Validation ayrımı (%80 Eğitim, %20 Doğrulama)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size]
)

# DataLoader yapıları
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==========================================
# 3. MLP MODEL MİMARİSİ (2 Gizli Katman + ReLU)
# ==========================================
class MNIST_MLP(nn.Module):

  def __init__(self):
    super(MNIST_MLP, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(28 * 28, 256)  # 1. Gizli Katman
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(256, 128)  # 2. Gizli Katman
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(128, 10)  # Çıkış Katmanı (10 sınıf)

  def forward(self, x):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x


model = MNIST_MLP().to(device)

# Loss ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. EĞİTİM VE DEĞERLENDİRME DÖNGÜSÜ
# ==========================================
epochs = 10
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

print("\n--- Model Eğitimi Başlıyor ---")
start_time = time.time()

for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  correct_train = 0
  total_train = 0

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    _, predicted = outputs.max(1)
    total_train += labels.size(0)
    correct_train += predicted.eq(labels).sum().item()

  epoch_train_loss = running_loss / len(train_loader.dataset)
  epoch_train_acc = correct_train / total_train

  # Validation aşaması
  model.eval()
  val_loss = 0.0
  correct_val = 0
  total_val = 0

  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_loss += loss.item() * images.size(0)
      _, predicted = outputs.max(1)
      total_val += labels.size(0)
      correct_val += predicted.eq(labels).sum().item()

  epoch_val_loss = val_loss / len(val_loader.dataset)
  epoch_val_acc = correct_val / total_val

  history["train_loss"].append(epoch_train_loss)
  history["train_acc"].append(epoch_train_acc)
  history["val_loss"].append(epoch_val_loss)
  history["val_acc"].append(epoch_val_acc)

  print(
      f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} - Train"
      f" Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} - Val Acc:"
      f" {epoch_val_acc:.4f}"
  )

print(f"\nToplam Eğitim Süresi: {time.time() - start_time:.2f} saniye")

# ==========================================
# 5. TEST SETİ DEĞERLENDİRMESİ
# ==========================================
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    total_test += labels.size(0)
    correct_test += predicted.eq(labels).sum().item()

test_accuracy = correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.4f}")

# Loss ve Accuracy grafiklerini göster
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Grafiği")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Grafiği")
plt.show()

# ==========================================
# 6. KENDİ EL YAZISI OKUL NUMARASI TESTİ (24120205061)
# ==========================================
folder_path = "gercek_el_yazisi_28x28"
okul_numarasi_str = "24120205061"

print("\n==============================================inton")
print("   KENDİ EL YAZISI OKUL NUMARASI TEST SONUÇLARI")
print("===============================================")

correct_custom = 0
model.eval()

with torch.no_grad():
  for i, gercek_rakam in enumerate(okul_numarasi_str):
    # Kaydettiğin dosya formatı: rakam_1_28x28.png vb.
    dosya_adi = f"rakam_{i+1}_28x28.png"
    dosya_yolu = os.path.join(folder_path, dosya_adi)

    if os.path.exists(dosya_yolu):
      image = Image.open(dosya_yolu).convert("L")
      tensor_image = TF.to_tensor(image)
      tensor_image = TF.normalize(tensor_image, (0.1307,), (0.3081,))
      tensor_image = tensor_image.unsqueeze(0).to(device)

      outputs = model(tensor_image)
      _, predicted = outputs.max(1)
      tahmin = predicted.item()

      durum = "DOĞRU" if str(tahmin) == gercek_rakam else "YANLIŞ"
      if str(tahmin) == gercek_rakam:
        correct_custom += 1

      print(
          f"{i+1}. Basamak -> Gerçek: {gercek_rakam} | Modelin Tahmini:"
          f" {tahmin} [{durum}]"
      )
    else:
      print(
          f"Uyarı: {dosya_yolu} dosyası bulunamadı! Klasör yolunu kontrol edin."
      )

basari_orani = (correct_custom / len(okul_numarasi_str)) * 100
print("----------------------------------------------")
print(
    f"Özel El Yazısı Başarı Oranı: %{basari_orani:.2f} ({correct_custom}/"
    f"{len(okul_numarasi_str)} doğru)"
)
print("===============================================")