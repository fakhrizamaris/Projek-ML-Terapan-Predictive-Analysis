# Laporan Proyek Machine Learning - Fakhri Djamaris

## Domain Proyek

Analisis prediktif harga saham merupakan aspek krusial dalam industri keuangan dan investasi. BMW (_Bayerische Motoren Werke AG_) sebagai salah satu produsen mobil premium terbesar di dunia memiliki pergerakan saham yang menarik untuk dianalisis. Prediksi harga saham dapat membantu investor dalam mengambil keputusan investasi yang lebih informed.

Dalam era digital saat ini, penggunaan machine learning untuk memprediksi harga saham telah menjadi pendekatan yang semakin populer karena kemampuannya dalam mengidentifikasi pola dan tren dari data historis.

Pada riset jurnal yang berjudul _"Stock Market Prediction with Historical Time Series Data and Sentimental Analysis of Social Media Data_" [[1]](https://ieeexplore.ieee.org/document/9121121) menunjukkan bahwa meskipun pasar saham cenderung efisien, masih terdapat pola-pola yang dapat diprediksi menggunakan teknik machine learning.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga penutupan (_Close price_) saham BMW berdasarkan data historis?
2. Seberapa akurat model _Machine learning_ dalam memprediksi harga saham BMW?
3. Fitur apa saja yang memiliki pengaruh signifikan terhadap prediksi harga saham?

### Goals

1. Mengembangkan model _Machine learning_ yang dapat memprediksi harga penutupan saham BMW
2. Mengukur dan membandingkan performa berbagai algoritma _Machine Learning_ untuk mendapatkan prediksi yang optimal
3. Mengidentifikasi dan menganalisis fitur-fitur yang berpengaruh terhadap pergerakan harga saham

### Solution Statements

1. Mengimplementasikan tiga algoritma _Machine Learning_:
   - _K-Nearest Neighbors (KNN)_
   - _Random Forest_
   - _Linear Regression_
2. Melakukan optimasi model dengan:
   - _Feature scaling_ menggunakan _StandardScaler_
   - _Hyperparameter_ tuning untuk model _Random Forest_
3. Menggunakan metrics MSE (_Mean Squared Error_) untuk evaluasi model

## Data Understanding

### Kondisi Data

1. _Missing Value_: Tidak ditemukan nilai yang hilang _(missing value_) pada dataset. Hal ini menunjukkan data lengkap untuk seluruh fitur.
2. Duplikasi Data: Tidak terdapat duplikat data setelah dilakukan pemeriksaan menggunakan metode _.duplicated()_.
3. _Outlier_: Pada fitur harga _(Close, Open, High, Low_), tidak ditemukan outlier ekstrem berdasarkan analisis visual menggunakan boxplot.
Namun, pada fitur Volume, terdapat beberapa nilai yang jauh lebih besar dibandingkan nilai rata-rata. Hal ini diantisipasi sebagai variasi normal karena volume perdagangan dapat berfluktuasi signifikan berdasarkan aktivitas pasar.
![outlier_volume](https://github.com/user-attachments/assets/d8d08ed6-f637-45b9-b587-50ee148f4aaf)


### Tautan Sumber Data
Dataset yang digunakan berasal dari Kaggle dan dapat diakses melalui tautan berikut:
_BMW Stock Data 1996-2024_ [[2]](https://www.kaggle.com/datasets/mhassansaboor/bmw-stock-data-1996-2024).


### Variabel-variabel pada dataset:

Dataset yang digunakan berisi informasi historis harga saham BMW dengan 7212 baris data dan 7 fitur. Berikut rincian fitur-fiturnya:

1. Date:
   - Tipe Data: object
   - Deskripsi: Tanggal perdagangan saham. Digunakan sebagai indeks dalam analisis data.
2. Open:
   - Tipe Data: Float
   - Deskripsi: Harga pembukaan saham pada hari perdagangan.
3. High:
   - Tipe Data: Float
   - Deskripsi: Harga tertinggi saham selama hari perdagangan.
4. Low:
   - Tipe Data: Float
   - Deskripsi: Harga terendah saham selama hari perdagangan.
5. Close:
   - Tipe Data: Float
   - Deskripsi: Harga penutupan saham pada akhir hari perdagangan.
   - Catatan: Fitur ini digunakan sebagai target (dependent variable) untuk model prediksi.
6. Adj Close:
   - Tipe Data: Float
   - Deskripsi: Harga penutupan yang disesuaikan, mencerminkan dividen atau aksi korporasi.
7. Volume:
   - Tipe Data: Integer
   - Deskripsi: Jumlah total lembar saham yang diperdagangkan selama hari perdagangan.

## Exploratory Data Analysis (EDA)

#### Ringkasan Statistik

Analisis statistik awal menunjukkan:

- Rentang harga saham (_Close_) dari €17.04 hingga €122.59
- Rata-rata harga _Close_ adalah €56.42

#### Analisis Tren Data

Visualisasi tren harga penutupan menunjukkan:
![tren-data](https://github.com/user-attachments/assets/b9c2ac05-e16f-46b0-b1cd-41a4c9ac6ebb)

- Tren kenaikan secara keseluruhan dalam harga saham BMW selama periode tersebut
- Beberapa fluktuasi harga yang signifikan, dengan penurunan tajam yang diikuti pemulihan
- Pergerakan harga yang relatif konsisten tanpa pencilan (outlier) yang ekstrem

#### Analisis Korelasi

Matriks korelasi mengungkapkan hubungan penting:
![matriks-korelasi](https://github.com/user-attachments/assets/531ed4f7-1dde-4090-9cb9-6c47f7d2d4c6)

1. Korelasi positif yang kuat (>0,99) antara:
   - Harga Close dan Open (0,99)
   - Harga Close dan High (0,99)
   - Harga Close dan Low (0,99)
2. Korelasi negatif antara Volume dan metrik harga:
   - Volume menunjukkan korelasi negatif lemah dengan semua fitur terkait harga
   - Hal ini menunjukkan volume perdagangan yang lebih tinggi mungkin sedikit berkorelasi dengan penurunan harga

Visualisasi pairplot mengkonfirmasi hubungan ini dan menunjukkan:
![korelasi-pairplot](https://github.com/user-attachments/assets/e06a3497-b225-4d90-856b-2a44ac8b7ad1)


- Hubungan linear antara semua fitur terkait harga
- Hubungan non-linear yang tersebar antara Volume dan fitur lainnya
- Pola pengelompokan yang jelas dalam pergerakan harga

## Data Preparation

1. _Handling Missing Values_:

   - Tidak terdapat _Missing Value_ pada data.

2. _Feature Engineering_:

   - Konversi kolom Date menjadi _datetime_
   - Penggunaan Date sebagai index untuk analisis _time series_

3. _Feature Selection_:

   - Menghilangkan fitur Volume karena memiliki korelasi yang lemah
   - Memilih _Close_ sebagai target variable

4. _Data Splitting_:

   - Train-test split dengan ratio 80:20

5. _Standardization_:
   - Menggunakan StandardScaler untuk normalisasi fitur numerik
   - _Scaling_ diterapkan pada fitur: _Adj_Close, High, Low, Open_

## Modeling

### 1. K-Nearest Neighbors (KNN)

- Prinsip algoritma: Memprediksi berdasarkan rata-rata k tetangga terdekat
- Hyperparameter yang digunakan:
  - n_neighbors=12
- Kelebihan:
  - Non-parametrik, bekerja baik dengan data non-linear
  - Sederhana untuk dipahami dan diimplementasikan
- Keterbatasan:
  - Sensitif terhadap skala fitur
  - Membutuhkan komputasi intensif untuk dataset besar

### 2. Random Forest

- Prinsip algoritma: Ensembel pohon keputusan menggunakan metode bagging
- Hyperparameter yang digunakan:
  - n_estimators=45
  - max_depth=32
  - random_state=321
  - n_jobs=-1
- Kelebihan:
  - Menangani hubungan non-linear dengan baik
  - Memberikan informasi pentingnya fitur
  - Lebih tahan terhadap _overfitting_
- Keterbatasan:
  - Membutuhkan komputasi intensif
  - Kurang dapat diinterpretasi dibanding model yang lebih sederhana

### 3. Linear Regression

- Prinsip algoritma: Memodelkan hubungan linear antara fitur dan target
- Kelebihan:
  - Sederhana dan mudah diinterpretasi
  - Efisien secara komputasi
  - Bekerja baik untuk hubungan linear
- Keterbatasan:
  - Mengasumsikan hubungan linear

## Evaluation

### Perbandingan Model dengan matriks MSE
| Model              | Train Error | Test Error |
|:------------------:|:-----------:|:----------:|
| KNN                | 0.000237    | 0.000272   |
| Random Forest      | 0.000031    | 0.000213   |
| Linear Regression  | 0.000191    | 0.000192   |


### Analisis Hasil

1. Evaluasi Keberhasilan Proyek:

   - Proyek berhasil membangun tiga model prediktif.
   - Random Forest menunjukkan kinerja terbaik dengan MSE terendah rata-rata dari data train dan test.
   - Semua model menunjukkan kemampuan prediksi yang wajar untuk harga saham BMW.

2. Pencapaian Tujuan:

   - Tujuan utama prediksi harga tercapai.
   - Model Random Forest menunjukkan performa sangat baik dengan MSE test.
   - Model mempertahankan kinerja baik pada data yang belum pernah dilihat.

3. Penyelesaian Masalah:
   - Berhasil menciptakan model prediksi harga saham yang andal.
   - Random Forest memberikan keseimbangan terbaik antara akurasi dan generalisasi.
   - Model dapat digunakan untuk prediksi harga jangka pendek dengan tingkat kepercayaan yang wajar.
   
4. Identifikasi Fitur Berpengaruh: Analisis dan pemilihan fitur menunjukkan bahwa beberapa variabel memainkan peran penting dalam memprediksi pergerakan harga saham:
   - Open: Harga pembukaan memiliki korelasi yang sangat kuat dengan harga penutupan (Close). Ini menunjukkan pola pergerakan yang konsisten dalam satu hari perdagangan.
   - High dan Low: Harga tertinggi dan terendah selama hari perdagangan memberikan informasi tentang volatilitas pasar yang memengaruhi harga penutupan.
   - Adj Close: Harga penutupan yang disesuaikan mempertimbangkan dividen dan aksi korporasi lainnya, sehingga menjadi indikator yang lebih akurat terhadap nilai aktual saham.
   - Volume: Walaupun memiliki korelasi yang lebih rendah dibandingkan fitur lain, volume perdagangan mencerminkan aktivitas pasar dan minat investor terhadap saham tertentu.


### Rekomendasi

1. Menggunakan Random Forest sebagai model utama karena kinerjanya yang superior
2. Menerapkan pelatihan ulang model secara berkala untuk mempertahankan akurasi prediksi
3. Menambahkan teknik feature engineering untuk potensial peningkatan performa model
4. Mempertimbangkan penambahan faktor eksternal (indeks pasar, sentimen berita) sebagai fitur

## Referensi

[1] M. Kesavan, J. Karthiraman, Rajadurai T. Ebenezer and S. Adhithyan, "Stock Market Prediction with Historical Time Series Data and Sentimental Analysis of Social Media Data", 2020 4th International Conference on Intelligent Computing and Control Systems (ICICCS), 2020, pp. 477-482.

[2] M. H. Saboor, "BMW Stock Data 1996-2024," Kaggle, 2024.
