# nagyadat

transformdata.py:
- 16-21. sor: szükséges adatok beolvasása .mat file-ból
- ImageDataset: DataSet megadása, itt lehet beállítani milyen módon legyen transzformálva a képek
- read_data: adatok szétválogatása: mennyi train, validation, test adat legyen
- 80-82. sor: dataloader-ek megadása
- mean_std: átlag, szórás kiszámolása a képekből a normalizáláshoz
- file végén beállítása a végleges transzformációknak a dataset-en illetve ezek elmentése

train.py:

- 57.sorig a dataloader-ek, dataset-ek betöltése
- 60-67.sor: eszköz beállítása a gpura, vgg16 model betöltése, plusz layerek beadása, 40 osztály legyen a végén (10-50 éves korig becsültem)
- train_model: 100 epoch-ban tanul a train, és valid dataloadereket használtam a train-re és validation-re.
