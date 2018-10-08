from src.methods.baseline import process_baseline
from src.methods.gan import process_gan

from sklearn.metrics import precision_score, recall_score, f1_score

def eval_cs_model(model, dev_gen, anchordata=None):
    true_labels = dev_gen.classes
    if anchordata is not None:
        predicted_labels = process_baseline(model, dev_gen, anchordata)
    else:
        predicted_labels = process_gan(model, dev_gen)

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print('precision: {} \n recall: {} \n f1 score: {} \n'.format(
        precision, recall, f1
    ))
    
    