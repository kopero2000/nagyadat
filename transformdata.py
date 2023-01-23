def read_data(size, validation_size, test_size):
    test_start = int(size-(test_size*size))
    valid_start = int(test_start-(validation_size*size))
    print(size, test_start, valid_start)
    data_train = ImageDataset('imdb', 0, valid_start)
    data_valid = ImageDataset('imdb', valid_start, test_start)
    data_test = ImageDataset('imdb', test_start, size)
    return data_train, data_valid, data_test


torch.save(dataloader_train_normal, "dataloader_train_normal.pth")
torch.save(dataloader_v_normal, "dataloader_v_normal.pth")
torch.save(dataloader_test_normal, "dataloader_test_normal.pth")

class ImageDataset(Dataset):
    def __init__(self, dir, start = 0, end = 10 ):
        self.img_labels = labels[start:end]
        self.dir = dir
        self.images = []
        
        for im in self.img_labels:
            imgpath = "/mnt/idms/home/a100/vizibela/data/imdb_crop/"+im[0]
            if path.exists(imgpath):
                image = PIL_image = PIL.Image.open(imgpath).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                self.images.append(transform(image)) #itt resize + totensor
        
        self.images = torch.stack(self.images)
        transform = transforms.Compose([])
    def settransform(self, transform):
        self.transform = transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):        
        img = self.transform(self.images[idx])
        return img, self.img_labels[idx][1]
