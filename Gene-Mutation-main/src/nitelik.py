import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

# Uyarıları bastır (gerekiyorsa gösterilebilir)
warnings.filterwarnings("ignore")

# 1. Veri Seti Yolu ve Hedef
csv_path = "final_merged_clinical_data_FULL.csv"
target_col = "Cancer Type"

# 2. Dosyayı Yükle
df = pd.read_csv(csv_path)
df = df.dropna(subset=[target_col])
y = LabelEncoder().fit_transform(df[target_col].astype(str))

# 3. Test Edilecek Öznitelik
feature = "Cancer Type Detailed"  # ← Burayı istediğin öznitelik için değiştir

# 4. Sütun Temizliği ve Encode
if df[feature].isnull().all():
    print(f"{feature} tamamen eksik. Atlaniyor.")
else:
    X = df[[feature]].dropna()
    y_aligned = y[X.index]

    # Label encode gerekiyorsa
    if X[feature].dtype == "object" or X[feature].dtype.name == "category":
        X[feature] = LabelEncoder().fit_transform(X[feature].astype(str))

    # 5. Eğitim/Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_aligned, test_size=0.2, stratify=y_aligned, random_state=42
    )

    # 6. Model Oluştur
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    # 7. Tahmin ve Değerlendirme
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 8. Özet Bilgi
    print(f"\n📊 Öznitelik: {feature}")
    print(f"Tip: {df[feature].dtype}")
    print(f"Benzersiz Değer Sayısı: {df[feature].nunique()}")
    print(f"Boş Değer Sayısı: {df[feature].isnull().sum()}")
    print(f"Kullanılan Veri Sayısı: {len(X)}")
    print(f"✅ Doğruluk Skoru: {acc:.4f}")
