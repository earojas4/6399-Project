import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, class_limit=None):
        # Load the annnotations csv file
        self.annotations = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

        # Since the imported annotations csv does not have labeled columns,. column names must be mnanually assigned.
        # The columns are the images name, the x and y coordinates for the bottom left corner of the bounding box and top right corner of the bounding box.
        self.annotations.columns = ['image_filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']

        # This dictionary and function below converts the images class labels from string to numeric values for easier processing by the model .
        self.class_to_index = {
            'Corn leaf blight': 0,
            'Banana Fusarium Wilt': 1,
            'Banana healthy': 2,
            'Cherry armillaria mellea': 3,
            'Cherry leaf healthy': 4,
            'Corn Gray leaf spot': 5,
            'Corn leaf healthy': 6,
            'Corn rust leaf': 7,
            'Peach Anarsia Lineatella': 8,
            'Peach leaf healthy': 9,
        }
        self.annotations['label'] = self.annotations['label'].map(self.class_to_index)

        # This checks to see if the class labels are within the dictionary specified above.
        if self.annotations['label'].isnull().any():
            missing_labels = self.annotations[self.annotations['label'].isnull()]
            raise ValueError(f"Some labels in the dataset do not match the class_to_index mapping. "
                             f"Missing labels: {missing_labels}")
        if class_limit is not None:
            self.annotations = self.limit_samples_per_class(class_limit)
  # Data in this set is limited for certain classes, so this part of the data loader is set up to provide a limit for specified classes. This ensures that the data is more balanced by udersampling the majorty classes.
    def limit_samples_per_class(self, class_limit):
        limited_annotations = []
        for label, limit in class_limit.items():
            class_data = self.annotations[self.annotations['label'] == label]
            limited_annotations.extend(class_data.sample(n=min(limit if limit is not None else float('inf'), len(class_data))).values.tolist())
        return pd.DataFrame(limited_annotations, columns=self.annotations.columns)
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        # Provide the path to the image and load it with PIL.Image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Fetch the mapped numbered class label and bounding box coordinates as a float value.
        label = self.annotations.iloc[idx, self.annotations.columns.get_loc('label')]
        bbox = self.annotations.iloc[idx, 1:5].values.astype(float)

        # This will apply any transformations specified to the images within the  set.
        if self.transform:
            image = self.transform(image)

        # Converts the label and bounding box coordinates to pytorch tensors.
        label = torch.tensor(label, dtype=torch.long)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, label, bbox
