# Printing out precision, recall and f1 scores for each class by this model
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    labels=list(range(len(selected_classes))),
    zero_division=0
    )
for i, class_name in enumerate(selected_classes):
    print(f"{class_name}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1 Score = {f1[i]:.2f}")
# Calculating and printing average precision, recall and F1 scores across all classes. Also printing overall accuracy of the model. 
precision_all = precision_score(true_labels, predicted_labels, average='macro')
recall_all = recall_score(true_labels, predicted_labels, average='macro')
f1_all = f1_score(true_labels, predicted_labels, average='macro')
accuracy_all = accuracy_score(true_labels, predicted_labels)

print(f"Average Precision for all classes: {precision_all:.2f}")
print(f"Average Recall for all classes: {recall_all:.2f}")
print(f"Average F1 Score for all classes: {f1_all:.2f}")
print(f"Overall Accuracy: {accuracy_all}")
