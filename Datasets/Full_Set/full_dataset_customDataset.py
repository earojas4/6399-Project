#Banana Healthy is limited to balance dataset
combined_test_dataset = CustomDataset(
    annotations_file='/content/Combined-1/test/_annotations.csv',
    img_dir='/content/Combined-1/test',
    transform=transform,
    class_limit= {0:250, 1:None, 2:None, 3:None, 4:None, 5:None, 6:None, 7: None, 8:None, 9:None}
)
train_dataset = CustomDataset(
    annotations_file='/content/Combined-1/train/_annotations.csv',
    img_dir='/content/Combined-1/train',
    transform=transform,
    class_limit={0:1000, 1:None, 2:None, 3:None, 4:None, 5:None, 6:None, 7: None, 8:None, 9:None})
