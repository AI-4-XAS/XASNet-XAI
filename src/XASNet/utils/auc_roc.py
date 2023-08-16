from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def auc(scores, contributions):
    """
    function to obtain the AUC values based on
    ground truth and feature attribution scores 

    Args:
        scores (_type_): Feature attribution scores
        contributions (_type_): Ground truth

    Returns:
        roc_auc_score: AUC value
        fpr: False positive rate 
        tpr: True positive rate
    """
    if not isinstance(scores, np.ndarray) and \
        isinstance(contributions, np.ndarray):
        scores = np.asarray(scores)
        contributions = np.asarray(contributions)
    
    contr_mean = np.mean(contributions)
    contr_labels = [1 if i > contr_mean else 0 \
                    for i in contributions]
    
    fpr, tpr, _ = roc_curve(contr_labels, scores)

    return roc_auc_score(contr_labels, scores), fpr, tpr


def plot_roc_curve(fpr, tpr, save=False):
    """
    plots the roc curve based of the probabilities
    """

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if save:
        plt.savefig('./auc_roc.png', dpi=300, bbox_inches='tight')
