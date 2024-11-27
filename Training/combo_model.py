aerial_model_path = '/content/drive/MyDrive/6399/alex_UAV_complete_model3.pth'
ground_model_path = '/content/drive/MyDrive/6399/alex_complete_model3.pth'
num_classes=10
# Load the pre-trained model weights
aerial_model = torch.load(aerial_model_path)
ground_model = torch.load(ground_model_path)
aerial_weights = aerial_model.state_dict()
ground_weights = ground_model.state_dict()
# Initializing a new empty model. This example has pretrained AlexNet weights loaded into an empty AlexNet. 
# Other architectures may be used, so long as the combined models and the empty model have the same architecture.
unified_model = models.alexnet(pretrained=False)
# Modifying head to classify images into 10 distinct classes. Different architectures may need different head structures
unified_model.fc = nn.Sequential(
        nn.Linear(9216, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
# Interpolation factor. Lambda weight of 0.5 results in the combined weights being averaged.
# By varying the lambda weight, the contribution of each model's weights toward the combined model weights can be changed. 
lambda_weight = .5 

unified_state_dict = unified_model.state_dict()

# Combining the two sets of weights
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

# Load the interpolated weights into the unified model
unified_model.load_state_dict(unified_state_dict)
