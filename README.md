# Clustering MovieLens 32M Streamlit

App ini adalah versi Streamlit dari notebook `Clustering Movielens 32M.ipynb`.

## Jalankan lokal

```powershell
cd "C:\Users\didas\Downloads\movielens_streamlit"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Saat tombol `Proses Dataset` ditekan, app akan mengunduh dataset MovieLens 32M dari GroupLens jika folder `data\ml-32m` belum tersedia.

## Pakai dataset lokal

Pilih `Folder lokal` di sidebar, lalu isi path folder yang berisi:

- `movies.csv`
- `ratings.csv`

Path juga boleh menunjuk ke parent folder yang memiliki subfolder `ml-32m`.

## Deploy ke Streamlit Community Cloud

1. Push folder ini ke repository GitHub.
2. Buat app baru di Streamlit Community Cloud.
3. Pilih repository tersebut.
4. Set main file path ke `app.py` jika isi folder ini menjadi root repo. Jika folder ini berada sebagai subfolder repo, gunakan path subfoldernya, misalnya `outputs/movielens_streamlit/app.py`.
5. Deploy.

Catatan: MovieLens 32M cukup besar. Jika deploy cloud terkena limit resource, jalankan di mesin lokal/VM, atau precompute hasil clustering menjadi CSV yang lebih kecil.
