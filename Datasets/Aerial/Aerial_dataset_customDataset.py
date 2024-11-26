#The dataset was divided into test and train splits in roboflow before importing. Here I am using the defined CustomDatset function to prepare each dataset for loading.
# Class 1 and 9 are undersampled to better balance the dataset.
train_dataset = CustomDataset(
    annotations_file='/content/plant-diseases-detection-aerial-4/train/_annotations.csv',
    img_dir='/content/plant-diseases-detection-aerial-4/train',
    transform=transform,
    class_limit={0: 800, 1: None, 2:None, 3:800,4: None, 5:None, 6:None,7:None, 8:800, 9:None}
)

test_dataset = CustomDataset(
    annotations_file='/content/plant-diseases-detection-aerial-4/test/_annotations.csv',
    img_dir='/content/plant-diseases-detection-aerial-4/test',
    transform=transform,
    class_limit={0: 200, 1: None, 2:None, 3:200,4: None, 5:None, 6:None,7:None, 8:200, 9:None}
)
