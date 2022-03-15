import numpy as np

def lrt_test_stat(xmpl, at, data):
    mle = np.column_stack([xmpl.prob_model_mhat(data),xmpl.prob_model_nuhat(data)])
    fixed = np.tile([at],(len(data),1))

    nuhathat = xmpl.getnuhathat(at, data)
    cmle = np.column_stack([fixed,nuhathat])

    a = np.array([xmpl.logpdf_prob_model(m,d) for m,d in zip(cmle,data)])
    b = np.array([xmpl.logpdf_prob_model(m,d) for m,d in zip(mle,data)])
    return -2*(a-b)

