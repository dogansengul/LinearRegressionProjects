import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Alternatif: 'Agg', 'Qt5Agg' veya 'MacOSX'
import matplotlib.pyplot as plt

"""
FEATURES:
    Nüfus (Population): Mağazanın bulunduğu şehrin ortalama nüfusu (örneğin bin kişi cinsinden).
    Hane Gelir Ortalaması (Average Household Income): Şehirdeki hane başına ortalama yıllık gelir (bin TL cinsinden).
    Reklam Bütçesi (Advertising Budget): Mağazanın yıllık reklam bütçesi (bin TL cinsinden).
    Rakip Mağaza Sayısı (Number of Competitors): Aynı şehirdeki (veya yakın civardaki) rakip mağaza sayısı.
    Şube Yaşı (Store Age): Şubenin faaliyete geçmesinden bu yana geçen süre (yıl cinsinden).

    Hedef (Target): Yıllık satış geliri (“bin TL” cinsinden).
"""

# ------------------------------
# Sentetik Veri Oluşturma
# ------------------------------
np.random.seed(42)
n_samples = 200

# 5 adet özellik oluşturuluyor:
X1 = np.random.randint(10, 1000, n_samples).astype(float)  # Nüfus (bin kişi)
X2 = np.random.randint(10, 100, n_samples).astype(float)  # Hane Gelir Ortalaması (bin TL)
X3 = np.random.randint(0, 50, n_samples).astype(float)  # Reklam Bütçesi (bin TL)
X4 = np.random.randint(1, 20, n_samples).astype(float)  # Rakip Mağaza Sayısı
X5 = np.random.randint(1, 30, n_samples).astype(float)  # Şube Yaşı (yıl)

# Gerçek (ground truth) katsayılar:
beta_0_true = 50
beta_1_true = 0.05
beta_2_true = 0.8
beta_3_true = 1.2
beta_4_true = -2
beta_5_true = 0.5

# Gürültü ekleyerek hedef değişkeni (Y) oluşturma:
noise = np.random.normal(0, 10, n_samples)
Y = (beta_0_true +
     beta_1_true * X1 +
     beta_2_true * X2 +
     beta_3_true * X3 +
     beta_4_true * X4 +
     beta_5_true * X5 +
     noise)

# Tüm özellikleri tek bir matris haline getiriyoruz: (n_samples, 5)
X = np.column_stack((X1, X2, X3, X4, X5))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
for i in range(10):
    print(f"{i + 1}. veri: X1:{X1[i]}, X2:{X2[i]}, X3:{X3[i]}, X4:{X4[i]}, X5:{X5[i]}, Y (Target): {Y[i]}")

# ------------------------------
# Veri Ölçeklendirme (Z-Score Normalization)
# ------------------------------
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_rescaled = (X - X_mean) / X_std

features = ["Nüfus", "Hane Gelir Ortalaması", "Reklam Bütçesi", "Rakip Mağaza Sayısı", "Şube Yaşı"]
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.scatter(X_rescaled[:, i], Y, alpha=0.6, edgecolors='black')
    plt.xlabel(features[i])
    plt.ylabel("Yıllık Satış Geliri")
    plt.grid(True)
plt.suptitle("Bağımsız Değişkenlerin Satış Geliri ile Scatter Plotları")
plt.tight_layout()
plt.show()


# ------------------------------
# Maliyet Fonksiyonu ve Gradient Hesaplama
# ------------------------------
def compute_cost(x_train, y_train, w, bias):
    """
    x_train: (m, n) özellik matrisi.
    y_train: (m,) hedef vektörü.
    w: (n,) ağırlık vektörü.
    bias: Skaler bias değeri.
    """
    m = x_train.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x_train[i], w) + bias
        cost += (f_wb_i - y_train[i]) ** 2
    return cost / (2 * m)


def compute_gradient(x_train, y_train, w, bias):
    """
    Her parametre için (w ve bias) maliyet fonksiyonunun türevini hesaplar.
    """
    m, n = x_train.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        err = (np.dot(x_train[i], w) + bias) - y_train[i]
        for j in range(n):
            dj_dw[j] += err * x_train[i, j]
        dj_db += err
    return dj_dw / m, dj_db / m


def gradient_descent(x_train, y_train, w_in, b_in, alpha, num_iters):
    """
    x_train: (m, n) özellik matrisi.
    y_train: (m,) hedef vektörü.
    w_in: (n,) başlangıç ağırlıkları.
    b_in: Başlangıç bias.
    alpha: Öğrenme oranı.
    num_iters: İterasyon sayısı.

    Her iterasyonda maliyeti kaydeder ve güncellenen (w, bias) değerlerini döndürür.
    """
    J_history = []
    w = w_in.copy()
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = compute_cost(x_train, y_train, w, b)
        J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")
    return w, b, J_history


# ------------------------------
# Gradient Descent Çalıştırma
# ------------------------------
# Ağırlık vektörünü (5,) boyutunda başlatıyoruz.
initial_w = np.zeros(5)
initial_b = 0.0
iterations = 1000
alpha = 0.03

w_final, b_final, J_hist = gradient_descent(X_rescaled, Y, initial_w, initial_b, alpha, iterations)
print(f"\nFinal parameters found by gradient descent:")
print(f"Bias: {b_final:0.2f}")
print(f"Weights: {w_final}")

# ------------------------------
# Iterasyon-Maliyet Grafiği
# ------------------------------
plt.figure(figsize=(8, 6))
plt.plot(range(len(J_hist)), J_hist, 'b-', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Iteration vs. Cost")
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    # Scatter plot: o özelliğin rescaled değeri ve Y
    plt.scatter(X_rescaled[:, i], Y, alpha=0.6, edgecolors='black', label="Data")
    # x ekseni için min - max aralığında 100 nokta alalım:
    x_vals = np.linspace(np.min(X_rescaled[:, i]), np.max(X_rescaled[:, i]), 100)
    # Diğer özellikler için 0 kabul ediyoruz (ortalama), bu nedenle:
    y_vals = b_final + w_final[i] * x_vals
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label="Prediction")
    plt.xlabel(features[i])
    plt.ylabel("Yıllık Satış Geliri")
    plt.title(f"{features[i]} Etkisi")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
