import numpy as np
import matplotlib.pyplot as plt
import torch
from wald_train import *
from wald_groundtruth import *
import scipy.stats as sps
from sklearn.isotonic import IsotonicRegression

def plot_data_showcase(xmpl, poi = 3.0, scale = 1.0):
    X,y,t = generate_data(xmpl, poi = poi, scale=scale, N = 1000)
    plt.scatter(X[:,0],X[:,1], c = y, alpha = 0.5)
    plt.xlim(*xmpl.prob_model_data1_range())
    plt.ylim(*xmpl.prob_model_data2_range())

def plot_profile(xmpl, model, calib_funcs, auto_rescale, obs_data = [0.0,0.0], axarr = None):
    xrange = xmpl.prob_model_poi_range()
    yrange = xmpl.prob_model_nuis_range()

    if axarr is None:
        f,axarr = plt.subplots(1,2)
        f.set_size_inches(10,5)

    obs_data = np.array([obs_data])
    mle = obs_data
    mu_scan = np.linspace(xrange[0],xrange[1])
    nuhathat = xmpl.getnuhathat(mu_scan,obs_data)
    profP = np.column_stack([mu_scan,nuhathat])

    mle = xmpl.prob_model_nuhat(obs_data[0])
    unprofP = np.column_stack([mu_scan,mle*np.ones(len(mu_scan))])

    NLLprof = -np.array([xmpl.logpdf_prob_model(p,obs_data) for p in profP])
    NLLunprof = -np.array([xmpl.logpdf_prob_model(p,obs_data) for p in unprofP])



    ax = axarr[0]
    grid = np.mgrid[xrange[0]:xrange[1]:101j,yrange[0]:yrange[1]:101j]
    Pi = np.swapaxes(grid,0,-1).reshape(-1,2)
    Li = np.array([xmpl.logpdf_prob_model(p,obs_data) for p in Pi])
    Li = Li.reshape(101,101).T
    ax.contour(grid[0],grid[1],-(Li-Li.max()), levels = 0.5*np.array([1,2,3])**2, colors = 'k')
    ax.plot(profP[:,0],profP[:,1],c = 'steelblue')
    ax.plot(unprofP[:,0],unprofP[:,1],c = 'grey')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_ylabel(r'$\nu$')
    ax.set_xlabel(r'$\mu$')

    ax = axarr[1]
    calib_fwd, calib_bwd = calib_funcs
    lrt_app_calib = calib_bwd.predict(np.array([parametrized_eval(xmpl,model,torch.Tensor(obs_data),p).detach().numpy()[0,0] for p in mu_scan]))
    lrt_app_auto = auto_rescale(np.array([parametrized_eval(xmpl,model,torch.Tensor(obs_data),p).detach().numpy()[0,0] for p in mu_scan]))

    lrt_stat_profiled = 2*(NLLprof - NLLprof.min())
    lrt_stat_unprofld = 2*(NLLunprof - NLLunprof.min())
    ax.plot(mu_scan,lrt_stat_unprofld, c = 'grey', label = 'unprofiled')
    ax.plot(mu_scan,lrt_stat_profiled, label = 'true profile LR')
    ax.plot(mu_scan,lrt_app_calib, label = 'trained profile LR (calib.)')
    ax.plot(mu_scan,lrt_app_auto, label = 'trained profile LR (uncalib.)')
    ax.hlines([1,4,9],xrange[0],xrange[1], colors = 'k')
    ax.set_xlim(xrange[0],xrange[1])
    ax.set_ylim(0,17)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$t_\mu(x)$')
    ax.legend()


def plot_trained_model_and_exdata(xmpl, model, null, alt, ax):
    poi_one = null[0]
    poi_two = alt[0]
    xrange = xmpl.prob_model_data1_range()
    yrange = xmpl.prob_model_data2_range()
    grid = np.mgrid[xrange[0]:xrange[1]:101j,yrange[0]:yrange[1]:101j]
    Xi = np.swapaxes(grid,0,-1).reshape(-1,2)
    yi_1 = parametrized_eval(xmpl,model,torch.Tensor(Xi).float(),[poi_one]).detach().numpy()
    yi_1 = yi_1.reshape(101,101).T

    yi_2 = parametrized_eval(xmpl,model,torch.Tensor(Xi).float(),[poi_two]).detach().numpy()
    yi_2 = yi_2.reshape(101,101).T

    levels = np.linspace(yi_1.min(), 0.7*yi_1.max(),3)
    c1 = ax.contour(grid[0],grid[1], yi_1, levels = levels, colors = 'steelblue', )
    c2 = ax.contour(grid[0],grid[1], yi_2, levels = levels, colors = 'maroon', )

    label1  = rf'$\mu={null[0]:.1f}$'
    label2  = rf'$\mu={alt[0]:.1f}$'

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.legend([
        c1.legend_elements()[0][0],
        c2.legend_elements()[0][0]
    ], [label1,label2], loc = 'upper left')


def plot_trained_model(xmpl, model, poi_left = 0.0, poi_right = 1.0):
    xrange = xmpl.prob_model_data1_range()
    yrange = xmpl.prob_model_data2_range()
    grid = np.mgrid[xrange[0]:xrange[1]:101j,yrange[0]:yrange[1]:101j]
    Xi = np.swapaxes(grid,0,-1).reshape(-1,2)
    yi_left = parametrized_eval(xmpl,model,torch.Tensor(Xi).float(),[poi_left]).detach().numpy()
    yi_left = yi_left.reshape(101,101).T


    yi_right = parametrized_eval(xmpl,model,torch.Tensor(Xi).float(),[poi_right]).detach().numpy()
    yi_right = yi_right.reshape(101,101).T

    f,axarr = plt.subplots(1,2)
    ax = axarr[0]
    ax.contourf(grid[0],grid[1], yi_left)
    ax.set_xlim(*xmpl.prob_model_data1_range())
    ax.set_ylim(*xmpl.prob_model_data2_range())


    _X,_y,_t = generate_data(xmpl, 100,1.5,poi_left)
    ax.scatter(_X[:,0],_X[:,1],c = _y, edgecolors='k')    


    ax = axarr[1]
    ax.contourf(grid[0],grid[1], yi_right)
    ax.set_xlim(*xmpl.prob_model_data1_range())
    ax.set_ylim(*xmpl.prob_model_data2_range())


    _X,_y,_t = generate_data(xmpl, 100,1.5,poi_right)
    ax.scatter(_X[:,0],_X[:,1],c = _y, edgecolors='k')    


    f.set_size_inches(10,5)    


def get_errs(a,b,cut):
    size = len(a[a>cut])/len(a)
    power = len(b[b>cut])/len(b)
    return size,power

def get_roc(a,b, range = (0,1)):
    edges = np.linspace(*range,1001)
    roc = np.array([get_errs(a,b,e) for e in edges])
    return roc[:,0],roc[:,1],edges
    
def auto_rescale_func(xmpl, model, null, alt):
    poi = null[0]
    _X,_y,_ = generate_fixed_ref(xmpl, 100000, null, alt)
    _p = parametrized_eval(xmpl,model,_X,[poi])
    _s,_w,_e = get_roc(_p[_y[:,0]==0][:,0],_p[_y[:,0]==1][:,0])

    def p_to_t(_p):
        pvals = np.interp(_p,_e,_s)
        _tc = sps.chi2(1).ppf(1-pvals)
        return _tc
    return p_to_t
    
def compare_roc_curves_to_asymptotics(xmpl, model, calib_funcs, auto_rescale, obs_data, null, alt):
    f,axarr = plt.subplot_mosaic('''
    ACGFF
    BDEFF
    ''')
    f.set_size_inches(15,5)
    X,y,t = generate_fixed_ref(xmpl, N = 10000, null=null, alt = alt)
    p = parametrized_eval(xmpl,model,torch.Tensor(X).float(),[t[0][0]]).detach().numpy()
    bins = np.linspace(0,1)


    lrt_p = lrt_test_stat(xmpl, t[0][0],X)
    tnull = lrt_p[y[:,0]==0]
    talt = lrt_p[y[:,0]==1]


    ax = axarr['A']
    ax.hist(p[y==0], density=True, histtype='step', bins = bins, label = 'trained null', edgecolor = 'steelblue');
    ax.hist(p[y==1], density=True, histtype='step', bins = bins, label = 'trained alt', edgecolor = 'maroon');
    ax.set_xlabel(r'$s_\phi(x;\mu)$')
    ax.set_ylim(1e-3,1e4)

    fpr_approx, tpr_approx, _ = get_roc(p[y==0],p[y==1], (0,1))
    fpr_lrt, tpr_lrt,_ = get_roc(tnull,talt,(0,30))

    ax.set_yscale('log')
    ax.legend()

    ax = axarr['B']
    non_centrality = xmpl.get_non_centrality(t[0],t[1])
    # non_centrality = lrt_test_stat(xmpl, t[0][0],np.array([[int(x) for x in xmpl.expcted_data([t[1][0],t[1][1]])]]))
    ats = np.linspace(0.001,0.999,101)

    asymp_null = sps.chi2(1)
    asymp_alt  = sps.ncx2(df = 1, nc = non_centrality)

    ti = np.linspace(0,25)
    ax.hist(tnull, bins = np.linspace(0,25), histtype = 'step', density=True, edgecolor = 'steelblue');
    ax.hist(talt, bins = np.linspace(0,25), histtype = 'step', density=True, edgecolor = 'maroon');
    ax.plot(ti,asymp_null.pdf(ti), label = r'$\chi^2$', color = 'steelblue')
    ax.plot(ti,asymp_alt.pdf(ti), label = r'$\chi^2_\mathrm{nc}(\Lambda^2)$', color = 'maroon')
    ax.set_xlabel(r'$t_{\mu}(x)$')
    ax.set_yscale('log')
    ax.set_ylim(1e-4,2e1)
    ax.legend()

    cuts = asymp_null.ppf(ats)
    size = 1-asymp_null.cdf(cuts)
    power = 1-asymp_alt.cdf(cuts)


    ax = axarr['C']
    ax.plot(fpr_approx,tpr_approx, label = 'trained')
    ax.plot(fpr_lrt,tpr_lrt, label = 'LRT', linestyle = 'dashed')
    ax.plot(size,power, c='r', label = 'aymptotics', linestyle = 'dotted')
    ax.set_xlabel('size')
    ax.set_ylabel('power')
    ax.legend()
        

    ax = axarr['D']
    ax.scatter(p[:,0],lrt_p, alpha = 0.2, label =r'$(s,t)$')
    rv = np.linspace(0,1)
    _, calib_bwd = calib_funcs
    iso_ti = calib_bwd.predict(rv)
    ax.plot(rv,iso_ti, c = 'r', label = 'iso. regression')
    ax.set_xlabel(r'$s(x,\mu)$')
    ax.set_ylabel(r'$t_\mu(x)$')
    ax.legend()


    plot_profile(xmpl, model, calib_funcs, auto_rescale, obs_data, axarr=[axarr['E'],axarr['F']])

    plot_trained_model_and_exdata(xmpl, model, null, alt, ax = axarr['G'])

def neural_neyman(xmpl, model, calib_funcs, obs_data):


    trs = []
    poi_range = xmpl.prob_model_poi_range()
    mu_scan = np.linspace(poi_range[0],poi_range[1],21)
    lrt = np.array([lrt_test_stat(xmpl,p,torch.Tensor([obs_data])) for p in mu_scan])
    lrt_app = np.array([parametrized_eval(xmpl,model,torch.Tensor([obs_data]),p).detach().numpy()[0,0] for p in mu_scan])


    trs_lrt = []
    for poi in mu_scan:
        X,y,t = generate_data_one_alt(xmpl, N = 10000, scale = 1.0, poi = poi)
        X = X[y[:,0] == 0]
        p_lrt = lrt_test_stat(xmpl,poi,X)
        p = parametrized_eval(xmpl,model,X,np.array([poi])).detach().numpy()[:,0]

        sorted_p = np.sort(p)
        levels = sps.chi2(1).cdf([1,4,9])
        tr = [sorted_p[int(cut)] for cut in levels*len(X)]
        trs.append(tr)

        sorted_p_lrt = np.sort(p_lrt)
        tr_lrt = [sorted_p_lrt[int(cut)] for cut in levels*len(X)]
        trs_lrt.append(tr_lrt)
    trs = np.array(trs)
    trs_lrt = np.array(trs_lrt)

    f,axarr = plt.subplots(1,2)
    ax =  axarr[0]
    calib_func, calib_inverse = calib_funcs
    ax.plot(mu_scan,calib_func.predict(np.array([[1,4,9]]*len(mu_scan)).ravel()).reshape(-1,3), c = 'k')
    ax.plot(mu_scan,calib_func.predict(trs_lrt.ravel()).reshape(-1,3), c = 'green')
    ax.plot(mu_scan,trs, c = 'red')
    ax.plot(mu_scan,lrt_app)
    ax.plot(mu_scan,calib_func.predict(lrt))
    ax.set_ylim(0,1)

    ax = axarr[1]
    ax.plot(mu_scan,np.array([[1,4,9]]*len(mu_scan)), c = 'k')
    ax.plot(mu_scan,trs_lrt, c = 'green')
    ax.plot(mu_scan,calib_inverse.predict(trs.ravel()).reshape(-1,3), c = 'red')
    ax.plot(mu_scan,calib_inverse.predict(lrt_app))
    ax.plot(mu_scan,lrt)

    
def calibrate_model(xmpl, model, null, alt):
    poi = null[0]
    X,y,t = generate_fixed_ref(xmpl, N = 10000, null=null, alt = alt)
    p = parametrized_eval(xmpl,model,torch.Tensor(X).float(),[poi]).detach().numpy()
    lrt_p = lrt_test_stat(xmpl,poi,X)  

    calib_fwd = IsotonicRegression().fit(lrt_p,p[:,0])
    calib_bwd = IsotonicRegression().fit(p[:,0],lrt_p)
    return calib_fwd, calib_bwd

def calibration_curve(xmpl, model, null, alt):
    poi = null[0]
    X,y,t = generate_fixed_ref(xmpl, N = 10000, null=null, alt = alt)
    p = parametrized_eval(xmpl,model,torch.Tensor(X).float(),[poi]).detach().numpy()
    lrt_p = lrt_test_stat(xmpl,poi,X)  

    calib_fwd, calib_bwd = calibrate_model(xmpl, model, null, alt)

    ti = np.linspace(0,25)
    rv = np.linspace(0,1)
    iso_pi = calib_fwd.predict(ti)
    iso_ti = calib_bwd.predict(rv)
    one_sigma_cut = calib_fwd.predict([1.0])[0]
    two_sigma_cut = calib_fwd.predict([4.0])[0]
    tre_sigma_cut = calib_fwd.predict([9.0])[0]
    print(one_sigma_cut,two_sigma_cut,tre_sigma_cut)

    f,axarr = plt.subplots(1,3)
    ax = axarr[0]
    ax.scatter(p,lrt_p)
    ax.plot(iso_pi,ti,c = 'r')

    ax = axarr[1]
    ax.scatter(lrt_p,p)
    ax.plot(iso_ti,rv, c = 'r')


    ax = axarr[2]
    ax.hist(lrt_p, density=True)

    f.set_tight_layout(True)
