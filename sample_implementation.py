# This is an example on how the code in each folder was combined to create a ResNet50 POV combined model. 
# More detailed explanations of the code are available within the Datasets and Metrics_and_testing folders

import os
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import warnings
from sklearn.metrics import confusion_matrix,  precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
from itertools import zip_longest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from google.colab import drive
drive.mount('/content/drive')
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="*************")
project = rf.workspace("project-4ohwz").project("combined-nf2nh")
version = project.version(1)
dataset = version.download("retinanet")

selected_classes = {
            'Corn leaf blight',
            'Banana Fusarium Wilt',
            'Banana healthy',
            'Cherry armillaria mellea',
            'Cherry leaf healthy',
            'Corn Gray leaf spot',
            'Corn leaf healthy',
            'Corn rust leaf',
            'Peach Anarsia Lineatella',
            'Peach leaf healthy'}

lass CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, class_limit=None):
        self.annotations = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.annotations.columns = ['image_filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
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
        if self.annotations['label'].isnull().any():
            missing_labels = self.annotations[self.annotations['label'].isnull()]
            raise ValueError(f"Some labels in the dataset do not match the class_to_index mapping. "
                             f"Missing labels: {missing_labels}")
        if class_limit is not None:
            self.annotations = self.limit_samples_per_class(class_limit)
    def limit_samples_per_class(self, class_limit):
        limited_annotations = []
        for label, limit in class_limit.items():
            class_data = self.annotations[self.annotations['label'] == label]
            limited_annotations.extend(class_data.sample(n=min(limit if limit is not None else float('inf'), len(class_data))).values.tolist())
        return pd.DataFrame(limited_annotations, columns=self.annotations.columns)
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, self.annotations.columns.get_loc('label')]
        bbox = self.annotations.iloc[idx, 1:5].values.astype(float)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, label, bbox
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
combined_test_dataset = CustomDataset(
    annotations_file='/content/Combined-1/test/_annotations.csv',
    img_dir='/content/Combined-1/test',
    transform=transform,
)
train_dataset = CustomDataset(
    annotations_file='/content/Combined-1/train/_annotations.csv',
    img_dir='/content/Combined-1/train',
    transform=transform)

aerial_model_path = '/content/drive/MyDrive/6399/resnet50_complete_model3_UAV.pth'
ground_model_path = '/content/drive/MyDrive/6399/resnet50_complete_model3.pth'

num_classes=10

aerial_model = torch.load(aerial_model_path)
ground_model = torch.load(ground_model_path)
aerial_weights = aerial_model.state_dict()
ground_weights = ground_model.state_dict()
unified_model = models.resnet50(pretrained=False)
unified_model.fc = nn.Sequential(
        nn.Linear(unified_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
lambda_weight = .5 

unified_state_dict = unified_model.state_dict()

for key in unified_state_dict.keys():
    if 'fc.' in key:  
        old_key = key.replace('fc.0', 'fc').replace('fc.3', 'fc')
        if old_key in aerial_weights and old_key in ground_weights:
            unified_state_dict[key] = (1 - lambda_weight) * aerial_weights[old_key] + lambda_weight * ground_weights[old_key]
        elif old_key in aerial_weights:
            unified_state_dict[key] = aerial_weights[old_key]
        elif old_key in ground_weights:
            unified_state_dict[key] = ground_weights[old_key]
    else: 
        if key in aerial_weights and key in ground_weights:
            unified_state_dict[key] = (1 - lambda_weight) * aerial_weights[key] + lambda_weight * ground_weights[key]
        elif key in aerial_weights:
            unified_state_dict[key] = aerial_weights[key]
        elif key in ground_weights:
            unified_state_dict[key] = ground_weights[key]
unified_model.load_state_dict(unified_state_dict)

test_loader = DataLoader(combined_test_dataset, batch_size=128, shuffle=False, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unified_model.parameters(), lr=0.0005)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unified_model = unified_model.to(device)
for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels, *_ = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = unified_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Fine-tuning completed.")
# (Loeber, 2022)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unified_model.to(device)
unified_model.eval()
true_labels = []
predicted_labels = []
with torch.no_grad():
    for images, labels, _ in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = unified_model(images)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=selected_classes, yticklabels=selected_classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    labels=list(range(len(selected_classes))),
    zero_division=0
    )
for i, class_name in enumerate(selected_classes):
    print(f"{class_name}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1 Score = {f1[i]:.2f}")

precision_all = precision_score(true_labels, predicted_labels, average='macro')
recall_all = recall_score(true_labels, predicted_labels, average='macro')
f1_all = f1_score(true_labels, predicted_labels, average='macro')
accuracy_all = accuracy_score(true_labels, predicted_labels)

print(f"Average Precision for all classes: {precision_all:.2f}")
print(f"Average Recall for all classes: {recall_all:.2f}")
print(f"Average F1 Score for all classes: {f1_all:.2f}")
print(f"Overall Accuracy: {accuracy_all}")
