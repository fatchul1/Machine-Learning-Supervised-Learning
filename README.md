# Machine Learning for Reduce Cancellation Rate

Banyaknya pembatalan pesanan hotel menjadi masalah yang menjadi perhatian tim management karena berhubungan langsung dengan revenue. Berdasarkan sumber dari dataset, tingkat cancellation rate mencapai 37%. Tingkat cancel paling tinggi terjadi pada tahun 2016, sebanyak 46% dari total cancel yang terjadi.

Membuat model machine learning untuk memprediksi apakah customer akan melakukan cancel atau tidak, sehingga pihak hotel dapat melakukan pendekatan dengan customer yang diprediksi akan cancel. Dengan demikian tingkat cancellation rate turun. Hal ini akan meningkatkan revenue berdasarkan simulasi yang diberikan.

![gambar_github](https://user-images.githubusercontent.com/116563315/205405720-8eb5eca1-a7e2-428a-a237-25cd9343aa1b.jpg)

## Project Organization
1. requirements.txt : requirements to run this model
2. exploratory_data_analysis.ipynb : data analis untuk memahami data
3. preprocessing.ipynb : tahapan preprocessing
4. model.ipynb : tahapan modelling
5. hotel_booking.csv : dataset awal
6. hotel_booking_after_preprocessing5.csv : dataset usai preprocessing
7. README.md : project report


## Prerequisites

1. Download data [here](https://www.kaggle.com/datasets/mojtaba142/hotel-booking)
2. Instalasi dengan `pip install requirements.txt`

## Getting Started
1. *Exploratory Data Analysis*, dilakukan proses cek duplikat, cek nilai null, cek tipe data sudah sesuai atau belum, mengecek deskripsi data apakah ada yang aneh atau tidak, cek data outlier, cek korelasi dan redundan.
2. *Preprocessing*, hasil dari pengamatan EDA selanjutnya diselesaikan di tahapan ini. Tahapan yang dilakukan pada proses ini yaitu *data cleansing, feature engineering, encoding, dan feature selection*
3. *Model*, pada proses ini dilakukan pembuatan model untuk supervised learning - klasifikasi. Model yang dilakukan uji diantaranya, logistic regression, k-nearest neighbor, decision tree, random forest, XGBoost, GaussianNB, catboost, dan adaboost. Dari hasil model tersebut, metriks yang dipilih ada recall. Berdasarkan pengamatan, dilakukan pemilihan model XGBoost.
4. Feature importance menggunakan SHAP value.

## Exploratory Data Analysis
Pada tahapan ini dilakukan pengamatan pada deskriptif statistik menggunakan df.info dan df.describe, hasilnya diketahui beberapa informasi. Selain itu, kami juga melakukan Univariate Analysis dan Multivariate Analysis. Sebelumnya, kami membagi dua jenis fitur, yaitu numerik dan kategorikal.

Pada univariate analysis, untuk jenis fitur numerikal, kami melakukan cek outlier dan cek distribusi. Sementara itu, pada fitur kategorikal, kami ingin melihat jumlahnya untuk setiap kelompok yang ada di dalamnya. Berikut hasilnya.

![cek outlier](https://user-images.githubusercontent.com/116563315/205411983-21cacdaa-394c-4cdc-8406-aa37ee40c773.jpg)

![cek distribusi](https://user-images.githubusercontent.com/116563315/205412093-9f1b2c24-452e-4d0a-bbb7-9c4f1c0c88e3.jpg)

![kategorikal](https://user-images.githubusercontent.com/116563315/205412305-398f9a04-35c6-465d-81e6-85b3a47c19a7.jpg)

Pada multivariate analysis, untuk jenis fitur numerikal kami melakukan perhitungan korelasi untuk mengetahui ada tidaknya fitur redundan. Selain itu, kami juga menampilkan pada fitur kategorikal hasil analisis untuk setiap hue cancel atau tidak. Berikut hasilnya.

![multivariate](https://user-images.githubusercontent.com/116563315/205418015-7ec2012b-b0a5-4fbb-91e9-233343b3bb88.jpg)

#### Business Insight
Salah satu hasil dari Exploratory Data Analysis (EDA) adalah munculnya business insight, beberapa yang didapat oleh kami sebagai berikut.

![BI_1](https://user-images.githubusercontent.com/116563315/205418237-099d3a53-b230-4429-894c-f866ca6c5846.jpg)

![BI_2](https://user-images.githubusercontent.com/116563315/205418274-f5b5e256-8eba-4cd5-a3d7-02f6bc2cc153.jpg)

![BI_3](https://user-images.githubusercontent.com/116563315/205418311-cc24a22d-b696-49bd-921c-98c487a15a40.jpg)

![BI_4](https://user-images.githubusercontent.com/116563315/205418338-8ea09f96-3284-4142-9383-4b26dfaa36cc.jpg)

## Data Preprocessing
#### Data Cleansing
Beberapa hal yang dilakukan pada data cleansing, yaitu:
1. Missing Values, dilakukan: drop 4 missing values pada kolom children; drop kolom company dan agent; mengisi missing values pada kolom country dengan mode.
2. Personal data, dilakukan drop 4 kolom data pribadi, yaitu name, email, phone number, credit_card.
3. Mengubah ‘undefined’ di meal, distribusi_channel, & market_segment.
4. Drop value adult yang bernilai 0.
5. Adjust data type, mengubah children menjadi int dan mengubah reserved_status_date menjadi datetime.
6. Handle outlier dengan metode z-score.

#### Feature Transformation
1. Melakukan log transformasi pada kolom lead_time, kolom ADR tidak dilakukan transformasi karena sudah mendekati normal.
2. Dilakukan normalisasi untuk kolom lead_time hasil transformasi, adr, serta total_of_special_requests.
3. Melakukan encoding dengan metode Labelling dan One Hot Encoding.
4. Karena target cukup proporsional, tidak dilakukan oversampling/undersampling.

#### Feature Extraction
1. reserved_room_type dan assigned_room_type digabung menjadi reserved_vs_assigned.
2. arrival_date_month diubah menjadi season.
3. country diubah menjadi origin_type.

## Model
Terdapat 8 kandidat model, yaitu Logistic Regression, KNN, Decision Tree, GaussiaNB, Random Forest, XGBoost, CatBoost, AdaBoost. Evaluasi dari 8 model tersebut menggunakan recall, bertujuan untuk menekan false negatif (diprediksi tidak cancel, ternyata iya). (diprediksi tidak cancel ternyata iya). Selain dapat menurunkan cancellation rate, tim bisnis juga dapat fokus kepada customer yang diprediksi false positif (diprediksi cancel ternyata tidak) sehingga cost untuk mempertahankan customer agar tidak cancel menjadi berkurang.

Hasil dari model terbaik berdasarkan metriks recall yaitu XGBoost, dengan nilai recall 0.75 serta nilai treshold roc_auc validation tidak lebih dari 10%.

Hasil dari XGBoost ini, kami lakukan Hyperparameter Tuning sehingga meningkatkan nilai recall menjadi 0.76. Perhatikan gambar.

![HT](https://user-images.githubusercontent.com/116563315/205428501-06e3cdca-23b0-450c-b320-8e58e66d10e7.jpg)

#### Feature Important
Dari beberapa fitur, berikut fitur yang paling menentukan prediksi dari model. 3 fitur paling berpengaruh adalah origin_type, market_segment_online_TA, deposit_type_no_deposit. Berikut tampilan semua fitur yang berpengaruh.


![feature_important](https://user-images.githubusercontent.com/116563315/205428677-fae70864-cc59-4e6b-9692-d317a2dbd04a.jpg)

## Business Recommendation
Beberapa business recommendation yang dapat meningkatkan revenue, diantaranya:

![BR_1](https://user-images.githubusercontent.com/116563315/205429430-f16121f8-6729-4880-aa72-0b41027f65fc.jpg)

![BR_2](https://user-images.githubusercontent.com/116563315/205429453-b60abe03-34d3-45f0-8d8f-9f56f7ef3833.jpg)

![BR_3](https://user-images.githubusercontent.com/116563315/205429475-7a908993-d694-4151-9197-e63a597f7809.jpg)

![BR_4](https://user-images.githubusercontent.com/116563315/205429509-9b885a6b-065e-4499-b6f9-3d529107ed02.jpg)

## Simulasi
Hasil dari pengaruh pembuatan Machine Learning dapat disimulasikan sebagai berikut. Terlihat pada flowchart dibawah, apabila suatu model memprediksi akan terjadi cancel ke suatu pelanggan, maka tim bisnis akan memberikan beberapa penawaran sehingga tidak terjadi cancel. Simulasi dari penambahan revenue dapat dilihat pada gambar selanjutnya.

![simulasi1](https://user-images.githubusercontent.com/116563315/205430891-846688c4-8f61-47c9-ab2b-3fdc8ea8c88f.jpg)

![simulasi2](https://user-images.githubusercontent.com/116563315/205430910-e28bb172-d4c3-4aad-8545-f6234ee96cb8.jpg)


Contact Me
https://www.linkedin.com/in/fatchul-arifin-197892135/

