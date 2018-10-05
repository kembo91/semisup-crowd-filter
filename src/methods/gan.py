import numpy as np

def process_gan(net, testdata):
    preds = net.predict_generator(testdata)
    td = np.where(preds >= 0.5)
    return [int(x) for x in td]