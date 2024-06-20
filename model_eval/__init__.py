import numpy as np
import pandas as pd
# Import model performance metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cf(pipeline,X_test,y_test):
    predictions = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    # Visualize confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Compute precision-recall curve
def model_scores(pipeline,X_test,y_test):
    predictions = pipeline.predict(X_test)
    model_report = classification_report(y_test, predictions)
    print(model_report)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    precision, recall, threshold = precision_recall_curve(y_test, y_prob)

    ap = average_precision_score(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f}, AP = {ap:.2f})', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# Assuming y_test are the true labels and y_score are the predicted probabilities
# Calculate ROC curve

def plot_roc(pipeline,X_test,y_test):
    predictions = pipeline.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)

    # Calculate area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Get predicted probabilities on the test set

def plot_gain_lift(pipeline,X_test,y_test):
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate gain and lift values
    def calculate_gain_lift(y_test, y_prob, num_buckets=10):
        order = np.argsort(y_prob)[::-1]
        y_test_sorted = y_test[order]
        y_prob_sorted = y_prob[order]
        total_positives = np.sum(y_test)
        
        gains = []
        lifts = []
        for i in range(1, num_buckets + 1):
            bucket_size = int(len(y_prob) * i / num_buckets)
            bucket_true = y_test_sorted[:bucket_size]
            bucket_positives = np.sum(bucket_true)
            gains.append(bucket_positives / total_positives)
            lifts.append(bucket_positives / (total_positives * (bucket_size / len(y_prob))))
            
        return gains, lifts

    gains, lifts = calculate_gain_lift(np.array(y_test), y_prob)

    # Plot the gain and lift charts
    num_instances = range(1, len(gains) + 1)

    plt.figure(figsize=(12, 5))

    # Gain Chart
    plt.subplot(1, 2, 1)
    plt.plot(num_instances, gains, marker='o')
    plt.title('Gain Chart')
    plt.xlabel('Number of Instances')
    plt.ylabel('Cumulative Gain')

    # Lift Chart
    plt.subplot(1, 2, 2)
    plt.plot(num_instances, lifts, marker='o')
    plt.title('Lift Chart')
    plt.xlabel('Number of Instances')
    plt.ylabel('Lift')

    plt.tight_layout()
    plt.show()

def model_eval_report(pipeline,X_test,y_test, name, cv):
    print ("Evaluation of {} model".format(name))
    print(pd.DataFrame(cv))
    plot_roc(pipeline,X_test,y_test)
    model_scores(pipeline,X_test,y_test)
    plot_cf(pipeline,X_test,y_test)
    plot_gain_lift(pipeline,X_test,y_test)