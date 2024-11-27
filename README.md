# View Based Training of Convolutional Neural Networks for Plant Disease Identification
##Project Description
Crop losses due to disease are estimated to cost the global economy $220 billion. By the year 2050, up to 50% more food must be produced to feed the world’s growing population. (Food and Agriculture Organization of the United Nations, 2024) Detecting diseases earlier could be key in this needed production increase. 
This research aims to explore the effects of combining aerial crop image data (macro-level) with ground-level images of leaves (micro-level) for the purposes of crop disease identification. By incorporating varying viewpoints, disease identification performance may be improved.
To achieve this, transfer learning was used to pretrain models with images from each point of view. Previous research has shown variability among widely used Convolutional Neural Network architectures(Tirkey et al. (2023)) (Roy & Bhaduri (2021)) (Jafar et al. (2024)). 
AlexNet (Krizhevsky et al., 2017), MobileNetv3(Howard et al., 2017), GoogleNet(Szegedy et al., 2015) and VGG19 (Simonyan & Zisserman, 2014) architectures were used in this project.
The compiled image dataset features healthy and diseased peach, corn, cherry and banana plants. After combining the weights of these POV models, their precision, recall and F1 scores are compared against models that were trained on the full data without concern for point of view. 

## Navigation
### Datasets
Data preperation and import code for each set

### Results
Metrics and confusion matrices for each model grouped by POV first then architecture

### Testing and metrics
Code used to test each model, generate confusion matrices and calculate metrics

### References
Full reference list for the project

