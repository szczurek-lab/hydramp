from keras import callbacks
import sklearn.metrics as metrics
import numpy as np

""" 
From Bioninja Challange:
https://github.com/jludwiczak/GrumpyPeptides 
"""


class ClassifierLogger(callbacks.Callback):

    def __init__(
            self,
            out_path='./',
            patience=10,
            out_fn='',

    ):
        self.f1 = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.no_improve = 0

    def on_epoch_end(self, epoch, logs={}):
        cv_pred = self.model.predict(self.validation_data[0], batch_size=1024)
        cv_true = self.validation_data[1]
        cv_pred_bin = np.where(cv_pred > 0.5, 1, 0)
        f1_val = metrics.f1_score(cv_true, cv_pred_bin)
        auc_val = metrics.roc_auc_score(cv_true, cv_pred)
        if self.f1 < f1_val:
            self.no_improve = 0
            print("Epoch %s - best F1: %s, AUC %s" % (epoch, round(f1_val, 4), round(auc_val, 4)))
            self.f1 = f1_val
            self.model.save(self.path + self.fn, overwrite=True)
        else:
            self.no_improve += 1
            print("Epoch %s - current F1: %s, AUC: %s" % (epoch, round(f1_val, 4), round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
        return
