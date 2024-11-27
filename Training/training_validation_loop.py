num_classes = 10
# Since the dataset was only divided into training and test sets, the training set will be divided into training and validation subsets.
# Since some classes in this set ar small, 10-fold validation may result in them being severely underepresented in some folds. 5 folds appear to strike a good balance.
k_folds = 5
dataset = train_dataset
kfold = KFold(n_splits=k_folds, shuffle=True)
fold_accuracies = []
fold_losses = []
num_epochs=30

# K-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"FOLD {fold+1}\n******************************************")

    # Split dataset into training and validation subsets and preparing dataloaders for each
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    #This example shows training and validation for a mobilenet V3 Small. Other models can be substituted here. The classifier head may be different for other architectures.
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    # This portion of the code isn't really necessary in a Google Colab environment or Jupyter Notebook, but if it is directly on a machine, the cuda device (usually a GPU) is selected if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Here the loss and optimization are specified for each epoch. Cross Entropy Loss is the most commonly used loss function for image classification models.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    #A scheduler is set up here to adjust the learning rate as training progresses
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Each fold is trained
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        #Epoch validation loss and accuracy are printed
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        scheduler.step()
    # Fold loss and accuracy are stored and printed
    fold_losses.append(val_loss)
    fold_accuracies.append(val_accuracy)

    print(f"Fold {fold+1} Finished. Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


# Average results after all epochs in all folds
print(f"Average Validation Loss: {np.mean(fold_losses):.4f}")
print(f"Average Validation Accuracy: {np.mean(fold_accuracies):.2f}%")
print("TRAINING COMPLETE")
