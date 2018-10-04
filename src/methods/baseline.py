from src.utils.utils import OutlierDetector
import numpy as np

def process_baseline(net, testdata, traindata):
    train_preds = net.predict_generator(traindata)
    outdet = OutlierDetector(train_preds, 8)
    connected = outdet.determine_connection(testdata)
    filtered = testdata[connected]
    return filtered
