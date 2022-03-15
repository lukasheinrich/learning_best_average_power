import numpy as np
import torch

def generate_fixed_ref(xmpl, N, null, alt):
    alts = [alt]
    null = np.array([null])
    
    theories = np.concatenate([null,alts])

    Xnull = xmpl.sample_prob_model(null[0],N) 
    Xalt0 = xmpl.sample_prob_model(alts[0],N) 


    ynull = np.zeros(N) 
    yalt0 = np.ones(N) 

    X = np.concatenate([Xnull,Xalt0])
    y = np.concatenate([ynull,yalt0])
    y = y.reshape(-1,1)

    return torch.Tensor(X).float(),torch.Tensor(y).float(), theories


def generate_data_one_alt(xmpl, N = 100, scale = 1, poi = 0.0, nuis_null = 1.0, mode = 'eval'):
    nuis_alt = np.random.uniform(*xmpl.prob_model_nuis_range(mode))
    alt = [poi-scale,nuis_alt]
    null = [poi,nuis_null]
    
    return generate_fixed_ref(xmpl, N, null, alt)
    
def generate_data(xmpl, N = 100, scale = 1, poi = 0.0, mode = 'eval'):
    nuis_null = np.random.uniform(*xmpl.prob_model_nuis_range(mode))
    alts = [
        [poi-1*scale,nuis_null],
        [poi+1*scale,nuis_null]
    ]
    null = np.array([[poi,nuis_null]])
    
    theories = np.concatenate([null,alts])

    Xnull = xmpl.sample_prob_model(null[0],N) 
    Xalt0 = xmpl.sample_prob_model(alts[0],N//2) 
    Xalt1 = xmpl.sample_prob_model(alts[1],N//2) 


    ynull = np.zeros(N) 
    yalt0 = np.ones(N//2) 
    yalt1 = np.ones(N//2) 

    X = np.concatenate([Xnull,Xalt0,Xalt1])
    y = np.concatenate([ynull,yalt0,yalt1])
    y = y.reshape(-1,1)

    return torch.Tensor(X).float(),torch.Tensor(y).float(), theories


def parametrized_eval(xmpl, model, X, pars):
    X = xmpl.preprocess(X)
    pcols = torch.Tensor(np.tile([pars],(len(X),1))).float()
    Xized = torch.cat([X,pcols],-1)
    return model(Xized)

def make_model(nfeats = 2, npars = 1):
    return torch.nn.Sequential(
        torch.nn.Linear(nfeats + npars,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100,1),
        torch.nn.Sigmoid(),
    )

def train(xmpl, Nsteps = 10000):
    model = make_model(nfeats = 2, npars=1)
    opt = torch.optim.Adam(model.parameters(), lr = 5e-5)

    losses = []
    for i in range(Nsteps):
        if i % 25 == 0:
            model.zero_grad()
        parloc = np.random.uniform(*xmpl.prob_model_poi_range('train'))
        scale  = np.random.uniform(*xmpl.prob_model_nuis_range('train'))
        X,y,t = generate_data(xmpl, poi = parloc, scale=scale, N = 1000, mode = 'train')
        p = parametrized_eval(xmpl, model,X,[parloc])
        l = torch.nn.functional.binary_cross_entropy(p,y)
        l.backward()
        if i % 1000 == 0:
            print(i,parloc,l)
        opt.step()
        losses.append(float(l))
    losses = np.array(losses)
    return model,losses