from src.methods.baseline import process_baseline
from src.methods.gan import process_gan

from sklearn.metrics import precision_score, recall_score, f1_score

def eval_cs_model(model, test_gen, anchordata=None):
    true_labels = test_gen.classes
    if anchordata is not None:
        predicted_labels = process_baseline(model, test_gen, anchordata)
    else:
        predicted_labels = process_gan(model, test_gen)

    precision = precision_score(predicted_labels, true_labels)
    recall = recall_score(predicted_labels, true_labels)
    f1 = f1_score(predicted_labels, true_labels)

    print('precision: {} \n recall: {} \n f1 score: {} \n'.format(
        precision, recall, f1
    ))
    
    