# nagyadat

transformdata.py:
- 16-21. sor: szükséges adatok beolvasása .mat file-ból
- ImageDataset: DataSet megadása, itt lehet beállítani milyen módon legyen transzformálva a képek
- read_data: adatok szétválogatása: mennyi train, validation, test adat legyen
- 80-82. sor: dataloader-ek megadása
- mean_std: átlag, szórás kiszámolása a képekből a normalizáláshoz
- file végén beállítása a végleges transzformációknak a dataset-en illetve ezek elmentése
