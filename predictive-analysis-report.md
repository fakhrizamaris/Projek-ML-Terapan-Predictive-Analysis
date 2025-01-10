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

Dataset yang digunakan adalah data historis saham BMW dari Kaggle [[2]](https://www.kaggle.com/datasets/mhassansaboor/bmw-stock-data-1996-2024) yang mencakup informasi harga harian dan volume perdagangan BWM dari tahun 1996-2024.

### Variabel-variabel pada dataset:

Dataset yang digunakan berisi informasi historis harga saham BMW dengan 1.258 baris data dan 7 fitur. Berikut rincian fitur-fiturnya:

1. Date: Tanggal perdagangan (digunakan sebagai index)
2. Open: Harga pembukaan saham
3. High: Harga tertinggi selama hari perdagangan
4. Low: Harga terendah selama hari perdagangan
5. Close: Harga penutupan saham
6. Adj Close: Harga penutupan yang disesuaikan
7. Volume: Jumlah lembar saham yang diperdagangkan

### Exploratory Data Analysis (EDA)

#### Ringkasan Statistik

Analisis statistik awal menunjukkan:

- Rentang harga saham (_Close_) dari €17.04 hingga €122.59
- Rata-rata harga _Close_ adalah €56.42

#### Analisis Tren Data

Visualisasi tren harga penutupan menunjukkan:
![](https://github.com/fakhrizamaris/Projek-ML-Terapan-Predictive-Analysis/blob/main/assets/tren-data.png)

- Tren kenaikan secara keseluruhan dalam harga saham BMW selama periode tersebut
- Beberapa fluktuasi harga yang signifikan, dengan penurunan tajam yang diikuti pemulihan
- Pergerakan harga yang relatif konsisten tanpa pencilan (outlier) yang ekstrem

#### Analisis Korelasi

Matriks korelasi mengungkapkan hubungan penting:
![Visualisasi Menggunakan Matriks Korelasi](https://github.com/fakhrizamaris/Projek-ML-Terapan-Predictive-Analysis/blob/main/assets/matriks-korelasi.png)

1. Korelasi positif yang kuat (>0,99) antara:
   - Harga Close dan Open (0,99)
   - Harga Close dan High (0,99)
   - Harga Close dan Low (0,99)
2. Korelasi negatif antara Volume dan metrik harga:
   - Volume menunjukkan korelasi negatif lemah dengan semua fitur terkait harga
   - Hal ini menunjukkan volume perdagangan yang lebih tinggi mungkin sedikit berkorelasi dengan penurunan harga

Visualisasi pairplot mengkonfirmasi hubungan ini dan menunjukkan:
![](https://github.com/fakhrizamaris/Projek-ML-Terapan-Predictive-Analysis/blob/main/assets/korelasi-pairplot.png)

- Hubungan linear antara semua fitur terkait harga
- Hubungan non-linear yang tersebar antara Volume dan fitur lainnya
- Pola pengelompokan yang jelas dalam pergerakan harga

## Data Preparation

1. _Handling Missing Values_:

   - Pengecekan menunjukkan tidak ada missing values dalam dataset

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
| Model           | Train Error | Test Error |
|------------------|-------------|------------|
| KNN             | 0.095       | 0.116      |
| Random Forest   | 0.002       | 0.024      |
| Linear Regression | 0.095     | 0.116      |


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

## Referensi

[1] M. Kesavan, J. Karthiraman, Rajadurai T. Ebenezer and S. Adhithyan, "Stock Market Prediction with Historical Time Series Data and Sentimental Analysis of Social Media Data", 2020 4th International Conference on Intelligent Computing and Control Systems (ICICCS), 2020, pp. 477-482.

[2] M. H. Saboor, "BMW Stock Data 1996-2024," Kaggle, 2024.
