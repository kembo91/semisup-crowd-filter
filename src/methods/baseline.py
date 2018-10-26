from src.utils.utils import OutlierDetector
import numpy as np

def process_baseline(net, testdata, anchordata):
    train_preds = net.predict_generator(anchordata)
    outdet = OutlierDetector(train_preds, 8)
    test_preds = net.predict_generator(testdata)
    connected = outdet.determine_connection(test_preds)
    return [int(x) for x in connected]
