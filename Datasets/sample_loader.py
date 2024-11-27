# The first arguement within the Pytorch DataLoader should be the customDataset that corresponds with the target split.
# All photos will be uploaded in batches of 128 images to ensure that there is sufficient RAM to process the set.
# Num workers is set to 0 when being run in a Colab environment. If run on a seperate machine, num workers can be increased to handle child operations if there is sufficient computational power.

test_loader = DataLoader(combined_test_dataset, batch_size=128, shuffle=False, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
