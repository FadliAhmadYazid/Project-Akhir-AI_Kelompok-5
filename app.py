import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Muat model yang telah disimpan
model = load_model('air_pollution_model.h5')

# Daftar kelas (pastikan ini sesuai dengan urutan kelas dalam model Anda)
classes = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Severe']

# Fungsi untuk memuat dan memproses gambar
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi
    return img_array

# Konfigurasi halaman dan header
st.set_page_config(page_title="Klasifikasi Polusi Udara", layout="centered", page_icon="🌍")
st.title("🌍 Klasifikasi Polusi Udara")
st.markdown("""
    Aplikasi ini menggunakan model **Deep Learning** untuk memprediksi tingkat polusi udara berdasarkan gambar yang Anda unggah.  
    Unggah gambar di bawah ini dan lihat hasil prediksinya!
""")

# Input gambar dari pengguna
uploaded_file = st.file_uploader("📤 Pilih gambar (format: jpg, png, jpeg)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    st.markdown("### 📷 Gambar yang Anda unggah:")
    img_path = 'temp/' + uploaded_file.name
    with open(img_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    img = preprocess_image(img_path)
    
    # Tampilkan gambar yang diunggah
    st.image(img_path, caption="Gambar yang diunggah", use_column_width=True)

    # Prediksi menggunakan model
    with st.spinner("🔍 Sedang memproses prediksi..."):
        predictions = model.predict(img)[0]  # Prediksi probabilitas untuk setiap kelas
        class_index = np.argmax(predictions)  # Indeks kelas dengan probabilitas tertinggi

    # Tampilkan hasil prediksi
    if class_index < len(classes):  # Pastikan indeks valid
        predicted_class = classes[class_index]
        st.success(f"✅ Prediksi Kelas: **{predicted_class}**")
    else:
        st.error("⚠️ Terjadi kesalahan pada prediksi.")
    
    # Diagram batang untuk distribusi probabilitas
    st.markdown("### 📊 Distribusi Probabilitas Prediksi:")
    fig, ax = plt.subplots()
    ax.bar(classes, predictions, color='skyblue')
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Probabilitas")
    ax.set_title("Probabilitas Prediksi untuk Setiap Kelas")
    ax.set_xticklabels(classes, rotation=90)
    st.pyplot(fig)
else:
    st.info("⚡ Silakan unggah gambar terlebih dahulu.")

# Footer
st.markdown("---")
