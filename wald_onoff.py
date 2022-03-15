import numpy as np
import scipy.stats as sps
from wald_groundtruth import lrt_test_stat

tau = 1
s0 = 15
b0 = 70

def preprocess(X):
    return (X-50.)/100.


def prob_model_data1_range():
    return [40,120]

def prob_model_data2_range():
    return [40,120]

def prob_model_poi_range(mode = 'eval'):
    if mode == 'eval':
        return [0,5]
    elif mode == 'train':
        return [-1,6]

def prob_model_nuis_range(mode = 'eval'):
    if mode == 'eval':
        return [0.7,1.3]
    elif mode == 'train':
        return [0.6,1.4]

def expcted_data(pars):
    mu,gamma = pars
    rates = np.array([mu*s0 + gamma * b0,tau*gamma*b0])
    return rates

def logpdf_prob_model(pars,data):
    rates = expcted_data(pars)
    return sps.poisson.logpmf(data,rates).sum(axis=-1)

def sample_prob_model(pars,N):
    return sps.poisson(expcted_data(pars)).rvs((N,2))
    
def getnuhathat(mu,data):
    n1,n2 = data[...,0],data[...,1]
    mu = mu*s0

    p = (-(n1+n2) + (1+tau)*mu)/(1+tau)
    q = -n2*mu/(1+tau)

    bhathat = -p/2 + np.sqrt(p**2/4 - q)
    gammahathat = bhathat/b0
    return gammahathat

def prob_model_mhat(data):
    n1,n2 = data[...,0],data[...,1]
    bhat = n2/tau
    muhat = n1 - bhat
    return muhat/s0

def prob_model_nuhat(data):
    n1,n2 = data[...,0],data[...,1]
    bhat = n2/tau
    gammahat = bhat/b0
    return gammahat

import sys
def get_non_centrality(a,b):
    return lrt_test_stat(sys.modules[__name__],a[0],np.array([[int(x) for x in expcted_data(b)]]))[0]