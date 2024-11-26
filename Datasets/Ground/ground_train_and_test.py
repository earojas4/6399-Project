#Banana Fusarium Wilt and Peach leaf healthy are limited to balance datasets
train_dataset = CustomDataset(
    annotations_file='/content/plant-diseases-detection-dataset-ground-3/train/_annotations.csv',
    img_dir='/content/plant-diseases-detection-dataset-ground-3/train',
    transform=transform,
    class_limit={0: None, 1: 800, 2:None, 3:None,4: None, 5:None, 6:None,7:None, 8:None, 9:800}
)

test_dataset = CustomDataset(
    annotations_file='/content/plant-diseases-detection-dataset-ground-3/test/_annotations.csv',
    img_dir='/content/plant-diseases-detection-dataset-ground-3/test',
    transform=transform,
    class_limit={0: None, 1: 200, 2:None, 3:None,4: None, 5:None, 6:None,7:None, 8:None, 9:200}
)
