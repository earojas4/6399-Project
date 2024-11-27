# Testing

*This step is performed after training*

POV models were tested on their respective subset training splits. The combined models and POV agnostic models were tested on the full data set’s testing split. (** see test_loop.py**)
Confusion matrices, precision, recall and F1 scores were generated for all classes. Average F1, precision, recall and accuracy for each model were also calculated. 
  
**confusion_matrix_plot.py** includes the code for generating a confusion matrix and using seaborn to visualize it.
**metrics.py** includes the code tp generate precision, recall and F1 scores both as a model average and per class.
