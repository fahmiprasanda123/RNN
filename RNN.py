import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Membaca data dari file Excel
file_path = 'data.xlsx'  # Ganti dengan path file Excel Anda
sheet_name = 'Cabai_Merah'  # Ganti dengan nama sheet yang diinginkan
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Menghapus kolom yang tidak diperlukan
df.drop(columns=['No', 'Komoditas (Rp)'], inplace=True)

# Transpose data sehingga tanggal menjadi index
df = df.transpose()
df.columns = df.iloc[0]  # Menggunakan baris pertama sebagai header
df = df[1:]  # Menghapus baris pertama karena sudah menjadi header

# Bersihkan spasi tambahan dalam string tanggal
df.index = df.index.str.replace(' ', '')

# Konversi index menjadi datetime
df.index = pd.to_datetime(df.index, format='%d/%m/%Y')

# Menghapus koma dari string angka dan mengonversinya menjadi tipe data numerik
df = df.applymap(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)

# Normalisasi data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Membuat dataset untuk pelatihan (menggunakan 3 data sebelumnya untuk memprediksi data ke-4)
def create_dataset(data, look_back=3):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 3
X, Y = create_dataset(scaled_data, look_back)

# Bentuk data untuk RNN
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Membangun model RNN
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(X.shape[2]))
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(X, Y, epochs=100, batch_size=1, verbose=0)

# Fungsi untuk memprediksi beberapa langkah ke depan
def predict_future(model, data, steps):
    predictions = []
    current_data = data[-look_back:]
    
    for _ in range(steps):
        current_data_reshaped = np.reshape(current_data, (1, look_back, data.shape[1]))
        predicted_scaled = model.predict(current_data_reshaped)
        predicted = scaler.inverse_transform(predicted_scaled)
        predictions.append(predicted[0])
        
        # Remove the first element from current_data and append the predicted_scaled
        current_data = np.append(current_data[1:], predicted_scaled, axis=0)
    return predictions

# Memprediksi data untuk beberapa langkah ke depan (misalnya 4 langkah ke depan)
predictions_future = predict_future(model, scaled_data, 4)
predictions_future_df = pd.DataFrame(predictions_future, columns=df.columns)

# Memilih hasil prediksi untuk tanggal yang diminta
predictions_future_df.index = ["02/01/2024", "03/01/2024", "04/01/2024", "05/01/2024"]

print(predictions_future_df)