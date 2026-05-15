import pandas as pd
import numpy as np
import re
import string
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.preprocessing import StandardScaler

def main():
    # --- 1. KONFIGURASI PATH ---
    # Menggunakan nama folder yang konsisten (Yoga-pratama)
    ROOT_DIR = 'Eksperimen_SML_Yoga-pratama'
    INPUT_DIR = os.path.join(ROOT_DIR, 'indonesian_crime_tweets_simulated_labeled_raw')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'preprocessing', 'indonesian_crime_tweets_simulated_labeled_preprocessed')

    input_file = os.path.join(INPUT_DIR, 'indonesian_crime_tweets_simulated_labeled.csv')
    output_file = os.path.join(OUTPUT_DIR, 'indonesian_crime_tweets_preprocessed.csv')

    # Memastikan folder output tersedia
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. LOAD DATA ---
    try:
        if not os.path.exists(input_file):
            print(f"[!] Error: File {input_file} tidak ditemukan.")
            return

        df = pd.read_csv(input_file)
        print(f"[*] Berhasil memuat: {input_file}")
    except Exception as e:
        print(f"[!] Error saat memuat file: {e}")
        return

    # --- 3. SETUP STOPWORDS & SLANG ---
    factory_sw = StopWordRemoverFactory()
    stop_words = set(factory_sw.get_stop_words())

    # Kata kunci yang harus dipertahankan
    removable = {'tidak', 'kurang', 'lama', 'lambat', 'sulit', 'sekali', 'masalah', 'salah'}
    for w in removable:
        stop_words.discard(w)

    additional_sw = {
        'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'kalo', 'amp', 'biar', 'bikin', 'nya',
        'ini', 'itu', 'saya', 'dan', 'di', 'si', 'ya', 'aja', 'ke', 'ka', 'pun',
        'halo', 'admin', 'min', 'mohon', 'woy', 'sih', 'loh', 'user', 'url'
    }
    stop_words.update(additional_sw)

    slang_dict = {
        "abis": "habis", "masi": "masih", "bgt": "sekali", "gak": "tidak",
        "ga": "tidak", "nggak": "tidak", "tdk": "tidak", "gk": "tidak",
        "banget": "sekali", "udah": "sudah", "sdh": "sudah", "pake": "pakai",
        "maling": "curi", "tkp": "lokasi"
    }

    # --- 4. PIPELINE FUNGSI ---
    def clean_process(text):
        # Basic Cleaning
        text = re.sub(r'@[A-Za-z0-9_]+|#|RT[\s]|http\S+|[0-9]+', '', str(text))
        text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()

        # Slang Handling & Stopwords
        words = text.split()
        fixed_words = []
        for w in words:
            word = slang_dict[w] if w in slang_dict else w
            if word not in stop_words and len(word) > 2:
                fixed_words.append(word)

        return ' '.join(fixed_words)

    # --- 5. EKSEKUSI ---
    print("[*] Memulai pembersihan teks...")
    if 'text' in df.columns:
        df['processed_text'] = df['text'].apply(clean_process)
        # Hapus baris kosong setelah cleaning
        df = df[df['processed_text'].str.strip() != ""]
    else:
        print("[!] Error: Kolom 'text' tidak ditemukan dalam dataset.")
        return

    print("[*] Melakukan standarisasi fitur numerik...")
    num_cols = ['user_followers', 'user_friends', 'retweet_count', 'favorite_count']
    # Pastikan kolom numerik ada sebelum scaling
    existing_num_cols = [col for col in num_cols if col in df.columns]

    if existing_num_cols:
        scaler = StandardScaler()
        df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])

    # --- 6. SIMPAN HASIL ---
    df.to_csv(output_file, index=False)
    print("-" * 30)
    print(f"[SUCCESS] File otomatis tersimpan di: {output_file}")
    print(f"[*] Total data akhir: {len(df)} baris")
    print("-" * 30)

if __name__ == "__main__":
    main()
