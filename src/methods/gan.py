import numpy as np

def process_gan(net, testdata):
    preds = net.predict_generator(testdata)
    td = np.where(preds >= 0.5)
    res = testdata[td]
    return res