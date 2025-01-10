# Laporan Analisis Prediktif Harga Saham BMW

## Pemahaman Data

Dataset yang digunakan berisi informasi historis harga saham BMW dengan 1.258 baris data dan 7 fitur. Berikut rincian fitur-fiturnya:

1. Date: Tanggal perdagangan (digunakan sebagai index)
2. Open: Harga pembukaan saham
3. High: Harga tertinggi selama hari perdagangan
4. Low: Harga terendah selama hari perdagangan
5. Close: Harga penutupan saham
6. Adj Close: Harga penutupan yang disesuaikan
7. Volume: Jumlah lembar saham yang diperdagangkan

### Analisis Eksplorasi Data (EDA)

#### Ringkasan Statistik
Analisis statistik awal menunjukkan:
- Rentang harga saham (Close) dari €56,70 hingga €102,45
- Rata-rata harga penutupan adalah €82,94
- Volume perdagangan menunjukkan variabilitas yang tinggi, berkisar antara 993.600 hingga 12.307.900 lembar saham

#### Analisis Deret Waktu
Visualisasi tren harga penutupan menunjukkan:
- Tren kenaikan secara keseluruhan dalam harga saham BMW selama periode tersebut
- Beberapa fluktuasi harga yang signifikan, dengan penurunan tajam yang diikuti pemulihan
- Pergerakan harga yang relatif konsisten tanpa pencilan (outlier) yang ekstrem

#### Analisis Korelasi
Matriks korelasi mengungkapkan hubungan penting:
1. Korelasi positif yang kuat (>0,99) antara:
   - Harga Close dan Open (0,99)
   - Harga Close dan High (0,99)
   - Harga Close dan Low (0,99)
2. Korelasi negatif antara Volume dan metrik harga:
   - Volume menunjukkan korelasi negatif lemah dengan semua fitur terkait harga
   - Hal ini menunjukkan volume perdagangan yang lebih tinggi mungkin sedikit berkorelasi dengan penurunan harga

Visualisasi pairplot mengkonfirmasi hubungan ini dan menunjukkan:
- Hubungan linear antara semua fitur terkait harga
- Hubungan non-linear yang tersebar antara Volume dan fitur lainnya
- Pola pengelompokan yang jelas dalam pergerakan harga

## Persiapan Data

1. Pembersihan Data:
   - Tidak ditemukan nilai yang hilang dalam dataset
   - Konversi kolom Date menjadi format datetime
   - Menetapkan Date sebagai index untuk analisis deret waktu

2. Pemilihan Fitur:
   - Menghapus Volume karena korelasi lemah dengan variabel target
   - Memilih Open, High, Low, dan Adj Close sebagai fitur
   - Harga Close ditetapkan sebagai variabel target

3. Pembagian Data:
   - 80% data pelatihan (1.006 sampel)
   - 20% data pengujian (252 sampel)

4. Standardisasi Fitur:
   - Menerapkan StandardScaler pada fitur numerik
   - Mentransformasi fitur agar memiliki rata-rata nol dan varians satu

## Pemodelan

### 1. K-Nearest Neighbors (KNN)
- Prinsip algoritma: Memprediksi berdasarkan rata-rata k tetangga terdekat
- Hyperparameter yang digunakan:
  - n_neighbors=9
- Kelebihan:
  - Non-parametrik, bekerja baik dengan data non-linear
  - Sederhana untuk dipahami dan diimplementasikan
- Keterbatasan:
  - Sensitif terhadap skala fitur
  - Membutuhkan komputasi intensif untuk dataset besar

### 2. Random Forest
- Prinsip algoritma: Ensembel pohon keputusan menggunakan metode bagging
- Hyperparameter yang digunakan:
  - n_estimators=50
  - max_depth=16
  - random_state=45
- Kelebihan:
  - Menangani hubungan non-linear dengan baik
  - Memberikan informasi pentingnya fitur
  - Lebih tahan terhadap overfitting
- Keterbatasan:
  - Membutuhkan komputasi intensif
  - Kurang dapat diinterpretasi dibanding model yang lebih sederhana

### 3. Regresi Linear
- Prinsip algoritma: Memodelkan hubungan linear antara fitur dan target
- Kelebihan:
  - Sederhana dan mudah diinterpretasi
  - Efisien secara komputasi
  - Bekerja baik untuk hubungan linear
- Keterbatasan:
  - Mengasumsikan hubungan linear
  - Sensitif terhadap pencilan

## Evaluasi

### Perbandingan Kinerja Model (MSE/1000)
```
                train    test
KNN             0,095   0,116
Random Forest   0,002   0,024
Linear Reg      0,095   0,116
```

### Analisis Hasil
1. Evaluasi Keberhasilan Proyek:
   - Proyek berhasil membangun tiga model prediktif
   - Random Forest menunjukkan kinerja terbaik dengan MSE terendah pada data train dan test
   - Semua model menunjukkan kemampuan prediksi yang wajar untuk harga saham BMW

2. Pencapaian Tujuan:
   - Tujuan utama prediksi harga tercapai
   - Model Random Forest menunjukkan performa sangat baik dengan MSE test 0,024
   - Model mempertahankan kinerja baik pada data yang belum pernah dilihat

3. Penyelesaian Masalah:
   - Berhasil menciptakan model prediksi harga saham yang andal
   - Random Forest memberikan keseimbangan terbaik antara akurasi dan generalisasi
   - Model dapat digunakan untuk prediksi harga jangka pendek dengan tingkat kepercayaan yang wajar

### Rekomendasi
1. Menggunakan Random Forest sebagai model utama karena kinerjanya yang superior
2. Menerapkan pelatihan ulang model secara berkala untuk mempertahankan akurasi prediksi
3. Menambahkan teknik feature engineering untuk potensial peningkatan performa model
4. Mempertimbangkan penambahan faktor eksternal (indeks pasar, sentimen berita) sebagai fitur

