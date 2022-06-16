
import pyreadr
path = '/Users/aubrey/Documents/SHK/Dropbox/Dingyi/Data/'

from utils.helper_function import generate_mse_fi


# Read original Data
data = pyreadr.read_r(path+'SRData.RData')
# path2 = path + 'mse&fi/'

dataname = ['abalone', 'bike', 'boston', 'concrete', 'cpu', 'csm', 'fb', 'parkinsons','servo', 'solar','synthetic1','synthetic2'] # real data
# dataname = ['boston'] # real data
######## use some of them

# Iterate to store ti and shap
mse_k0_from_scratch_inbag = []
mse_k0_from_scratch_oob = []
fi_k0_simulation_s = {}

mse_k1_from_scratch_inbag = []
mse_k1_from_scratch_oob = []
fi_k1_simulation_s = {}

# train&test
for index, name in enumerate(dataname):
    
    mse_inbag, mse_oob, fi_k0_simulation_s['df_{}'.format(name)] = generate_mse_fi(data[name], k=0, n_trees=200)
    mse_k0_from_scratch_inbag.append(mse_inbag)
    mse_k0_from_scratch_oob.append(mse_oob)

    mse_inbag, mse_oob, fi_k1_simulation_s['df_{}'.format(name)] = generate_mse_fi(data[name], k=1, n_trees=200)
    mse_k1_from_scratch_inbag.append(mse_inbag)
    mse_k1_from_scratch_oob.append(mse_oob)
    
