#Preparing a confusion matrix for the predicted vs true class values.
selected_classes = {
            'Corn leaf blight',
            'Banana Fusarium Wilt',
            'Banana healthy',
            'Cherry armillaria mellea',
            'Cherry leaf healthy',
            'Corn Gray leaf spot',
            'Corn leaf healthy',
            'Corn rust leaf',
            'Peach Anarsia Lineatella',
            'Peach leaf healthy'}
cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=selected_classes, yticklabels=selected_classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
