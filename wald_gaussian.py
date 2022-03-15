import numpy as np
import scipy.stats as sps

def preprocess(X):
    return X

def prob_model_data1_range():
    return [-5,5]

def prob_model_data2_range():
    return [-5,5]

def prob_model_poi_range(mode = 'eval'):
    if mode == 'eval':
        return [-3,3]
    elif mode == 'train':
        return [-5,5]

def prob_model_nuis_range(mode = 'eval'):
    if mode == 'eval':
        return [-3,3]
    elif mode == 'train':
        return [-5,5]


CORR = 0.8
COV =  np.array([[1.0,CORR],[CORR,1.0]])
COVINV = np.linalg.inv(COV)

def __prob_model(pars):
    return sps.multivariate_normal(mean = pars,cov =COV)

def getnuhathat(at, data):
    covinv = np.linalg.inv(COV)
    nuhathat = covinv[0,1]/covinv[1,1]*(data[:,0]-at)+data[:,1]
    return nuhathat

def get_non_centrality(a,b):
    return (a - b)**2/COV[0,0]

def sample_prob_model(pars, N):
    return __prob_model(pars).rvs(N)

def logpdf_prob_model(pars,data):
    return __prob_model(pars).logpdf(data)



def prob_model_mhat(data):
    n1,n2 = data[...,0],data[...,1]
    return n1

def prob_model_nuhat(data):
    n1,n2 = data[...,0],data[...,1]
    return n2

def expcted_data(pars):
    return pars

def get_non_centrality(a,b):
    return (a[0] - b[0])**2/COV[0,0]
