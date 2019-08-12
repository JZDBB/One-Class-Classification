from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

ran = np.linspace(0, 10, 11)
for i in ran:
    print(i)

# y_true = [0, 1, 1, 0, 1, 1]
# y_pred = [0, 1, 1, 0, 0, 1]
#
# acc =accuracy_score(y_true, y_pred)
# r = recall_score(y_true, y_pred, average='binary')
# p = precision_score(y_true, y_pred, average='binary')
# f1 = f1_score(y_true, y_pred, average='binary')
# print(acc, r, p, f1)