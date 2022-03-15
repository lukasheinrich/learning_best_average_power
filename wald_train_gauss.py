import wald_gaussian as xmpl
from wald_train import *
from wald_plots import *
from wald_groundtruth import *

model,losses = train(xmpl, 50000)
torch.save(model,'wald_gauss.ckpt')
