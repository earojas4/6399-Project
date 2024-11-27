# Training and Validation
All models were coded in python, using the Pytorch library, and trained within a Google Colab environment(on a T4 GPU High RAM runtime). The predefined architectures used for this project (AlexNet, VGG19, mobileNet V3 small, GoogleNet and ResNet50) were loaded in with their default weights (with layers not frozen to allow training). The fully connected heads were modified to classify the dataset into 10 distinct classes. Adam was used as the optimizer and Cross Entropy Loss was the loss function used for all models. A Step LR (step size= 10) was used to adjust the learning rate as validation progressed.

## Single POV Models and Full Set Models
The single POV models and the POV agnostic models were trained and validated using K-Fold validation, with k=5, for 30 epochs. For each fold the training dataset was split into training and validation sets and then put into data loaders with a batch size of 64 to ensure that not all data is loaded at once (possibly resulting in a session crash). (see **training_validation_loop.py** )

Single POV models were trained and validated only on their respective image subsets and saved for later use in a combined model. (see **sample_save.py**)
## Combined Models

POV models (of the same architecture) were imported, and their weight dictionaries were isolated. These weight dictionaries were combined with a lambda weight of 0.5 and added to an empty weight dictionary. While this functions the same as averaging the model weights, the lambda combination was selected to allow for testing with different contributions from each model. In testing for these models 0.5 had the best results. This combined dictionary is then loaded into an empty model of the same predefined architecture. The new unified model head is modified to classify the images into 10 distinct classes. Ten epochs of fine-tuning training are done on the combined model to ensure the head is functioning correctly. (see **combo_model.py**)

