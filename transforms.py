# Transform images to ensure they are loaded in a uniform manner. In this case, I am ensuring all images are 224x224 in the RGB space.
# The image data is also converted to a tensor and normalized to be within the expected range for the pretrained models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
