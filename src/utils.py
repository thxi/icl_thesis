import warnings
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


# a function to get precision/recall/AUC/F1 from y_true and y_pred
def get_metrics(y_true, y_pred):
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    res["precision"] = precision
    res["recall"] = recall
    res["auc"] = auc
    res["f1"] = f1
    return res
