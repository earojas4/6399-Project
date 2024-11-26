# The first arguement within the Pytorch DataLoader should be the customDataset that corresponds with the target split.
# Batch size should be adjusted based on available RAM. 
# num_workers can be increased if the machine this code is run on can handle child operations while main operations are being run.
test_loader = DataLoader(combined_test_dataset, batch_size=128, shuffle=False, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
