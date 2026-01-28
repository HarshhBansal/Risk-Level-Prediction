from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocessing import X_train_resampled, X_test, y_train_resampled, y_test


if __name__ == "__main__":
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    # Accuracy & report
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.title("Random Forest Confusion Matrix"); plt.show()

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(7,6))
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc:.2f})')
    plt.plot([0,1], [0,1], 'k--'); 
    plt.title("Random Forest ROC Curve"); 
    plt.legend(); 
    plt.show()
