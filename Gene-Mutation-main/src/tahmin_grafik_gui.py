import tkinter as tk
from tkinter import ttk
from collections import Counter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Sınıf isimleri ve tahmin çıktısı (örnek)
class_labels = [
    "AML", "Adrenocortical Carcinoma", "Bladder Urothelial Carcinoma", "Cervical SCC",
    "Cholangiocarcinoma", "Colon Adenocarcinoma", "DLBCLNOS", "Diffuse Glioma",
    "Esophageal Adenocarcinoma", "Glioblastoma", "HCC", "HGSOC", "LUAD", "LUSC",
    "NSGCT", "PAAD", "PLMESO", "PRAD", "Rectal Adenocarcinoma", "Renal Cell Carcinoma",
    "SKCM", "SOFT", "Stomach Adenocarcinoma", "THPA", "THYM", "UCEC", "UCS", "Uveal Melanoma"
]
preds = [0, 1, 1, 2, 3, 3, 3, 5, 5, 5, 5, 7, 8, 8, 8, 9, 9, 25, 25, 25, 3, 3, 3, 5, 5, 6, 6, 10, 10, 10, 11, 12]

def calculate_distribution(preds, labels):
    counter = Counter(preds)
    total = sum(counter.values())
    counts = [counter.get(i, 0) for i in range(len(labels))]
    percentages = [c / total * 100 for c in counts]
    return counts, percentages

def update_plot(selection):
    counts, percentages = calculate_distribution(preds, class_labels)
    data = [(label, c, p) for label, c, p in zip(class_labels, counts, percentages) if c > 0]

    if selection.startswith("Top"):
        n = int(selection.split()[1])
        data = sorted(data, key=lambda x: x[1], reverse=True)[:n]
    elif selection.startswith("Bottom"):
        n = int(selection.split()[1])
        data = sorted(data, key=lambda x: x[1])[:n]

    if not data:
        ax.clear()
        ax.set_title("Veri bulunamadı.")
        canvas.draw()
        return

    labels, values, pct = zip(*data)
    ax.clear()
    ax.bar(labels, values, color="#8FBEDC")

    for i, (label, value, percent) in enumerate(zip(labels, values, pct)):
        ypos = value + 0.1 if value < 2 else value * 0.98
        ax.text(i, ypos, f"%{percent:.1f}", ha='center', fontsize=8, fontweight="bold", color="navy")

    ax.set_title("Test Seti Tahmin Dağılımı (Temiz Çizim)", fontsize=12)
    ax.set_ylabel("Tahmin Edilen Sayı")
    ax.set_xlabel("Kanser Türü")
    ax.tick_params(axis='x', rotation=75)
    fig.tight_layout()
    canvas.draw()

# --- GUI Başlat ---
root = tk.Tk()
root.title("Tahmin Dağılımı Filtreleyici")

# --- ComboBox ---
options = ["Tüm Sınıflar", "Top 3", "Top 5", "Top 10", "Bottom 3", "Bottom 5"]
selected_option = tk.StringVar(value="Tüm Sınıflar")
combo = ttk.Combobox(root, textvariable=selected_option, values=options, state="readonly")
combo.pack(pady=10)
combo.bind("<<ComboboxSelected>>", lambda e: update_plot(selected_option.get()))

# --- Grafik Alanı (canvas sadece bir kez oluşturuluyor) ---
fig = Figure(figsize=(12, 6))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# İlk grafik çizimi
update_plot("Tüm Sınıflar")

# Uygulama döngüsü
root.mainloop()
