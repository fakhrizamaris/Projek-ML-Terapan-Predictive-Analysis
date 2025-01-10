# Laporan Proyek Machine Learning - Fakhri Djamaris

## Domain Proyek

Analisis prediktif harga saham merupakan aspek krusial dalam industri keuangan dan investasi. BMW (*Bayerische Motoren Werke AG*) sebagai salah satu produsen mobil *premium* terbesar di dunia memiliki pergerakan saham yang menarik untuk dianalisis. Prediksi harga saham dapat membantu investor dalam mengambil keputusan investasi yang lebih *informed*.

Dalam era digital saat ini, penggunaan *machine learning* untuk memprediksi harga saham telah menjadi pendekatan yang semakin populer karena kemampuannya dalam mengidentifikasi pola dan tren dari data historis.

Penelitian [1] menunjukkan bahwa meskipun pasar saham cenderung efisien, masih terdapat pola-pola yang dapat diprediksi menggunakan teknik *machine learning*.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga penutupan (*Close price*) saham BMW berdasarkan data historis?
2. Seberapa akurat model *machine learning* dalam memprediksi harga saham BMW?
3. Fitur apa saja yang memiliki pengaruh signifikan terhadap prediksi harga saham?

### Goals

1. Mengembangkan model *machine learning* yang dapat memprediksi harga penutupan saham BMW
2. Mengukur dan membandingkan performa berbagai algoritma *machine learning* untuk mendapatkan prediksi yang optimal
3. Mengidentifikasi dan menganalisis fitur-fitur yang berpengaruh terhadap pergerakan harga saham

### Solution Statements

1. Mengimplementasikan tiga algoritma *machine learning*:
   - *K-Nearest Neighbors* (KNN)
   - *Random Forest*
   - *Linear Regression*
2. Melakukan optimasi model dengan:
   - *Feature scaling* menggunakan *StandardScaler*
   - *Hyperparameter tuning* untuk model *Random Forest*
3. Menggunakan metrik MSE (*Mean Squared Error*) untuk evaluasi model

## Data Understanding

Dataset yang digunakan adalah data historis saham BMW [2] yang mencakup informasi harga harian dan volume perdagangan BWM dari tahun 1996-2024. Dataset ini memiliki beberapa fitur utama:

[Bagian selanjutnya sama seperti sebelumnya, hanya dengan penambahan format *italic* untuk kata asing]

## Referensi

[1] S. Joshi, D. Ranjan and V. S. Rajput, "Stock Market Prediction with Historical Time Series Data and Sentimental Analysis of Social Media Data," 2020 4th International Conference on Electronics, Communication and Aerospace Technology (ICECA), Coimbatore, India, 2020, pp. 422-427, doi: 10.1109/ICECA49313.2020.9121121.

[2] M. H. Saboor, "BMW Stock Data 1996-2024," Kaggle, 2024. [Online]. Available: https://www.kaggle.com/datasets/mhassansaboor/bmw-stock-data-1996-2024. [Accessed: Jan. 10, 2025].

