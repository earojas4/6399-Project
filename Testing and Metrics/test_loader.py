# A dataloader is prepared for the test dataset. All photos will be uploaded in batches of 128 images to ensure that there is sufficient RAM to process the set.
# Num workers is set to 0 when being run in a Colab environment. If run on a seperate machine, num workers can be increased to handle child operations if there is sufficient computational power.
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
