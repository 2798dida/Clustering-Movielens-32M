# Clustering MovieLens 32M

## Jalankan lokal

```powershell
cd ""
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
