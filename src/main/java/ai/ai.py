
import torch
import os
import argparse
import sys
import pandas as pd
import numpy as np
import importlib.util
from peptdeep.settings import global_settings
import sys


#torch.set_num_threads(36)
#torch.set_num_interop_threads(36)
#os.environ["OMP_NUM_THREADS"] = "36"
#os.environ["MKL_NUM_THREADS"] = "36"

def train_ms2(in_dir:str,
              use_valid=True,
              out_dir="./",
              out_prefix="ms2_pred",
              device='gpu',
              instrument='Eclipse',
              nce=27,
              mode_type="general",
              use_grid_nce_search=False,
              log_transform=False):
    from peptdeep.pretrained_models import ModelManager
    import numpy as np
    import pandas as pd
    import math
    pd.options.mode.chained_assignment = None  # default=‘warn’
    a = pd.read_csv(in_dir+"/psm_pdv.txt",sep="\t")

    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
        model_mgr.load_installed_models('generic')
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)
        model_mgr.load_installed_models('phos')
    model_mgr.train_verbose = True
    model_mgr.thread_num = 36
    model_mgr.epoch_to_train_ms2 = 20
    model_mgr.warmup_epoch_to_train_ms2 = 10
    model_mgr.batch_size_to_train_ms2 = 512
    #model_mgr.lr_ms2 = 0.0001
    model_mgr.lr_to_train_ms2 = 0.0001
    model_mgr.psm_num_per_mod_to_train_ms2 = 50
    model_mgr.top_n_mods_to_train = 10
    n_test_ms2 = np.min([1000,int(math.ceil(a.shape[0]*0.1)-10)])
    model_mgr.psm_num_to_test_ms2 = n_test_ms2
    model_mgr.psm_num_to_train_ms2 = a.shape[0] - n_test_ms2
    print("The number of PSMs to train MS2 model: ", model_mgr.psm_num_to_train_ms2)
    print("The number of PSMs to test MS2 model: ", model_mgr.psm_num_to_test_ms2)
    model_mgr.out_dir = out_dir
    model_mgr.nce = nce
    model_mgr.instrument = instrument
    model_mgr.use_grid_nce_search = use_grid_nce_search
    
    a["sequence"] = a["peptide"]
    a["nAA"] = a["sequence"].str.len()
    a.dtypes
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    b = pd.read_csv(in_dir+"/fragment_intensity_df.tsv",sep="\t",engine="pyarrow")
    if log_transform:
        print("log transform intensity data ...")
        b = b.apply(lambda x: np.log10(x+1)/np.log10(2))
    if use_valid:
        valid = pd.read_csv(in_dir+"/fragment_intensity_valid.tsv",sep="\t",engine="pyarrow")
    else:
        valid = pd.DataFrame(0, index=range(b.shape[0]), columns=b.columns)

    if device == 'cpu':
        ## get the number of cpus
        n_cpu = os.cpu_count()
        torch.set_num_threads(n_cpu)
    model_mgr.train_ms2_model(psm_df=a,matched_intensity_df=b,matched_valid_intensity_df=valid)
    model_mgr.ms2_model.save(out_dir+"/ms2_model.pt")
    return model_mgr


def train_rt(in_dir:str, out_dir:str, mode_type="general",device='gpu'):
    import pandas as pd
    import numpy as np
    import math
    from peptdeep.pretrained_models import ModelManager
    pd.options.mode.chained_assignment = None  # default=‘warn’
    a = pd.read_csv(in_dir+"/rt_train_data.tsv",sep="\t")
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
        model_mgr.load_installed_models('generic')
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)
        model_mgr.load_installed_models('phos')
    
    model_mgr.train_verbose = True
    model_mgr.thread_num = 36
    model_mgr.epoch_to_train_rt_ccs=40
    model_mgr.warmup_epoch_to_train_rt_ccs = 10
    #model_mgr.warmup_epoch_to_train_ms2 = 10
    model_mgr.batch_size_to_train_rt_ccs = 1024
    #model_mgr.lr_ms2 = 0.0001
    model_mgr.lr_to_train_rt_ccs = 0.0001
    model_mgr.top_n_mods_to_train = 10
    model_mgr.psm_num_per_mod_to_train_rt_ccs = 50
    n_test_rt = np.min([1000,int(math.ceil(a.shape[0]*0.1)-10)])
    model_mgr.psm_num_to_train_rt_ccs = a.shape[0] - n_test_rt
    model_mgr.psm_num_to_test_rt_ccs = n_test_rt
    model_mgr.out_dir = out_dir
    print("The number of peptides to train RT model: ", model_mgr.psm_num_to_train_rt_ccs)
    print("The number of peptides to test RT model: ", model_mgr.psm_num_to_test_rt_ccs)
    
    a.dtypes
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    # a['rt_norm'] = a['apex_rt']
    if device == 'cpu':
        ## get the number of cpus
        n_cpu = os.cpu_count()
        torch.set_num_threads(n_cpu)
    model_mgr.train_rt_model(psm_df=a)
    model_mgr.rt_model.save(out_dir+"/rt_model.pt")
    return model_mgr


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_models_from_docker():
    model_zip = os.path.join(global_settings['PEPTDEEP_HOME'], "pretrained_models/pretrained_models.zip")
    if os.path.exists(model_zip):
        print("The pre-trained models:" + model_zip)
    elif os.path.exists("/data/peptdeep/pretrained_models/pretrained_models.zip"):
        global_settings['PEPTDEEP_HOME'] = "/data/peptdeep"
        print("The pre-trained models: /data/peptdeep/pretrained_models/pretrained_models.zip")
    else:
        print("Will download pre-trained models from github.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict MS2 and RT.')
    parser.add_argument('--in_dir', default='./', help='Training data directory')
    parser.add_argument('--out_dir', default='./', help='Output directory')
    parser.add_argument('--out_prefix', default="test", help='Output prefix')
    parser.add_argument('--device', default='gpu', help='Device to use for prediction')
    parser.add_argument('--instrument', default='Eclipse', help='For MS2 prediction')
    parser.add_argument('--nce', default=27, help='For MS2 prediction')
    parser.add_argument('--tf_type', default="all", help='For library generation: rt, ms2, or all')
    parser.add_argument('--mode', default="general", help='general or phosphorylation')
    ## add log transform for intensity data
    parser.add_argument('--log_transform', action='store_true', help='log transform intensity data')
    parser.add_argument('--seed', default=2024, help='Random seed for training')

    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    out_prefix = args.out_prefix
    device = args.device
    instrument = args.instrument
    nce = float(args.nce)
    tf_type = args.tf_type

    set_seed(int(args.seed))

    if device == 'cpu':
        ## get the number of cpus
        n_cpu = os.cpu_count()
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(n_cpu)
        os.environ["OMP_NUM_THREADS"] = str(n_cpu)
        os.environ["MKL_NUM_THREADS"] = str(n_cpu)

    package_name = 'peptdeep'
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"The path of the {package_name} package is: {spec.origin}")
    else:
        print(f"The {package_name} package is not installed.")

    check_models_from_docker()

    if tf_type == "all" or tf_type == "test":
        if tf_type == "test":
            print("Test mode ...")
        model_mgr_rt = train_rt(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,device=device)
        model_mgr = train_ms2(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,log_transform=args.log_transform,nce=nce,device=device,instrument=instrument)
    elif tf_type == "rt":
        model_mgr_rt = train_rt(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,device=device)
    elif tf_type == "ms2":
        model_mgr = train_ms2(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,log_transform=args.log_transform,nce=nce,device=device,instrument=instrument)
