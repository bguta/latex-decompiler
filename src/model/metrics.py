import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import segmentation_models as sm

def focal(y_true, y_pred):
    return sm.metrics.f1_score(y_true, y_pred)
