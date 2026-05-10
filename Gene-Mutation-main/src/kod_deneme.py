import matplotlib.pyplot as plt
import seaborn as sns

# Eğitim verisindeki sınıf dağılımı
train_class_counts = df_train['Class'].value_counts().sort_index()
class_names = df_train['Cancer Type'].cat.categories

plt.figure(figsize=(12,6))
sns.barplot(x=class_names, y=train_class_counts.values)
plt.xticks(rotation=90)
plt.title("Eğitim Seti Sınıf Dağılımı")
plt.xlabel("Kanser Türü")
plt.ylabel("Örnek Sayısı")
plt.tight_layout()
plt.show()
