import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import sys
import importlib.util
import sys
import re


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
              log_transform=False,
              threads=1,
              verbose=1):
    from peptdeep.pretrained_models import ModelManager
    import numpy as np
    import pandas as pd
    import math
    pd.options.mode.chained_assignment = None  # default=‘warn’
    a = pd.read_csv(in_dir+"/psm_pdv.txt",sep="\t",dtype={'mod_sites': str, 'mods': str})

    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
        model_mgr.load_installed_models('generic')
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)
        model_mgr.load_installed_models('phos')
    model_mgr.train_verbose = True
    model_mgr.thread_num = threads
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
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)
    b = pd.read_csv(in_dir+"/fragment_intensity_df.tsv",sep="\t",engine="pyarrow")
    if log_transform:
        print("log transform intensity data ...")
        b = b.apply(lambda x: np.log10(x+1)/np.log10(2))
    if use_valid and os.path.exists(in_dir+"/fragment_intensity_valid.tsv"):
        print("Use valid intensity data ...")
        valid = pd.read_csv(in_dir+"/fragment_intensity_valid.tsv",sep="\t",engine="pyarrow")
    else:
        valid = pd.DataFrame(0, index=range(b.shape[0]), columns=b.columns)

    if device == 'cpu':
        # CPU-only tuning: enforce pools during training
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=threads, user_api="openmp"):
                torch.set_num_threads(threads)
                if verbose >=2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                model_mgr.train_ms2_model(psm_df=a,matched_intensity_df=b,matched_valid_intensity_df=valid)
    else:
        # GPU mode
        model_mgr.train_ms2_model(psm_df=a,matched_intensity_df=b,matched_valid_intensity_df=valid)
    model_mgr.ms2_model.save(out_dir+"/ms2_model.pt")
    return model_mgr


def train_rt(in_dir:str, out_dir:str, mode_type="general",device='gpu',threads=1,verbose=1):
    import pandas as pd
    import numpy as np
    import math
    from peptdeep.pretrained_models import ModelManager
    pd.options.mode.chained_assignment = None  # default=‘warn’
    a = pd.read_csv(in_dir+"/rt_train_data.tsv",sep="\t",dtype={'mod_sites': str, 'mods': str})
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
        model_mgr.load_installed_models('generic')
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)
        model_mgr.load_installed_models('phos')
    
    model_mgr.train_verbose = True
    model_mgr.thread_num = threads
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
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=threads, user_api="openmp"):
                torch.set_num_threads(threads)
                if verbose >=2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                model_mgr.train_rt_model(psm_df=a)
    else:
        model_mgr.train_rt_model(psm_df=a)
    model_mgr.rt_model.save(out_dir+"/rt_model.pt")
    # Save test results if available
    if hasattr(model_mgr.rt_model, 'test_df') and model_mgr.rt_model.test_df is not None:
        model_mgr.rt_model.test_df.to_csv(os.path.join(out_dir, "rt_test.tsv"), sep="\t", index=False)
    return model_mgr

def train_ccs(in_dir:str, out_dir:str, mode_type="general",device='gpu',threads=1,verbose=1):
    import pandas as pd
    import numpy as np
    import math
    from peptdeep.pretrained_models import ModelManager
    pd.options.mode.chained_assignment = None  # default=‘warn’
    a = pd.read_csv(in_dir+"/ccs_train_data.tsv",sep="\t",dtype={'mod_sites': str, 'mods': str})
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
        model_mgr.load_installed_models('generic')
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)
        model_mgr.load_installed_models('phos')

    model_mgr.train_verbose = True
    model_mgr.thread_num = threads
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
    print("The number of peptides to train CCS model: ", model_mgr.psm_num_to_train_rt_ccs)
    print("The number of peptides to test CCS model: ", model_mgr.psm_num_to_test_rt_ccs)

    a.dtypes
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    # a['rt_norm'] = a['apex_rt']

    #from alphabase.peptide.precursor import (
    #    refine_precursor_df,
    #    update_precursor_mz
    #)
    #if 'precursor_mz' not in a.columns:
    #    update_precursor_mz(a)
    if device == 'cpu':
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=threads, user_api="openmp"):
                torch.set_num_threads(threads)
                if verbose >=2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                model_mgr.train_ccs_model(psm_df=a)
    else:
        model_mgr.train_ccs_model(psm_df=a)
    model_mgr.ccs_model.save(out_dir+"/ccs_model.pt")
    return model_mgr

def set_seed(seed):
    import numpy as np
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

# =========================
# ENV setup BEFORE numpy/pandas/torch
# =========================
def configure_env_for_device(device: str, n_physical: int):
    """
    Must run BEFORE importing numpy/pandas/torch.

    CPU:
      - Use physical cores (often best on dual-socket / HT-heavy machines)
    GPU:
      - Keep CPU-side thread pools smaller to reduce overhead/oversubscription
    """
    device = (device or "gpu").lower()

    # If user already set these from the command line, respect them.
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(n_physical))
        os.environ.setdefault("MKL_NUM_THREADS", str(n_physical))
    else:
        # GPU mode: you can tune this; 2-8 is typically reasonable.
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")

    # Keep other libs from creating extra pools
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Optional: reduce OpenMP spinning
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_PROC_BIND", "true")

    # Optional: avoid torch.compile / inductor on Windows if you don't have MSVC "cl".
    # If you *do* have cl and want compilation, set TORCHDYNAMO_DISABLE=0 in env before running.
    if os.name == "nt":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def print_thread_state(tag: str = ""):
    if tag:
        print(f"\n==== {tag} ====")
    print("Python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("device mode:", device)
    print("n_physical:", n_physical, "n_logical:", n_logical)
    print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
    print("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS"))
    print("torch intra", torch.get_num_threads())
    print("torch interop", torch.get_num_interop_threads())
    try:
        print("mkldnn enabled:", bool(getattr(torch.backends, "mkldnn", None) and torch.backends.mkldnn.enabled))
    except Exception:
        pass
    try:
        import pprint
        pprint.pp(threadpool_info())
    except Exception:
        pass


def plot_rt(out_dir):
    import os
    import json
    import pandas as pd
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import scipy.stats
        from sklearn.metrics import median_absolute_error, r2_score
    except ImportError:
        print("Required libraries for plotting (matplotlib, sklearn, scipy) not found.")
        return

    test_file = os.path.join(out_dir, "rt_test.tsv")
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Skipping plotting.")
        return

    meta_file = os.path.join(out_dir, "meta.json")
    rt_min = 0.0
    rt_max = 1.0
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r') as f:
                content = f.read()
                # Robust parsing: handle unescaped backslashes commonly found in Windows paths in malformed JSON
                # This doubles backslashes that are not already escaped (neither preceded nor followed by a backslash)
                # and are not escaping a quote.
                import re
                # Use a lambda to avoid backslash escape issues in re.sub replacement string
                content = re.sub(r'(?<!\\)\\(?![\\"])', lambda m: r'\\', content)
                try:
                    meta_data = json.loads(content)
                except json.JSONDecodeError as je:
                    print(f"JSON decode failed even after regex fix: {je}")
                    # Ultimate fallback: just replace all single backslashes with double ones 
                    # except for escaped quotes
                    f.seek(0)
                    content = f.read()
                    content = content.replace('\\', '\\\\').replace('\\\\"', '\\"')
                    meta_data = json.loads(content)
                
                if meta_data:
                    # Collect all rt_min and rt_max values
                    all_rt_min = []
                    all_rt_max = []
                    for key in meta_data:
                        if isinstance(meta_data[key], dict):
                            if 'rt_min' in meta_data[key]:
                                all_rt_min.append(meta_data[key]['rt_min'])
                            if 'rt_max' in meta_data[key]:
                                all_rt_max.append(meta_data[key]['rt_max'])
                    
                    if all_rt_min:
                        rt_min = min(all_rt_min)
                    if all_rt_max:
                        rt_max = max(all_rt_max)
                        
                    print(f"Using global RT normalization factors: rt_min={rt_min}, rt_max={rt_max}")
        except Exception as e:
            print(f"Error reading meta.json: {e}")

    try:
        df = pd.read_csv(test_file, sep="\t")
        if 'rt_norm' not in df.columns or 'rt_pred' not in df.columns:
            print(f"Columns 'rt_norm' or 'rt_pred' not found in {test_file}")
            return
            
        # Denormalize to minutes
        y_obs = df['rt_norm'] * (rt_max - rt_min) + rt_min
        y_pred = df['rt_pred'] * (rt_max - rt_min) + rt_min

        cor = scipy.stats.pearsonr(y_obs, y_pred)[0]
        mae = median_absolute_error(y_obs, y_pred)
        r2 = r2_score(y_obs, y_pred)
        # The minimum delta RT which covers 95% of the data when sorted
        # i.e., the 95th percentile of the absolute difference between observed and predicted RT
        range_95 = np.percentile(np.abs(y_obs - y_pred), 95)

        plt.rcParams['figure.figsize'] = [5, 5]
        plt.scatter(y_obs, y_pred, s=4, c="blue", alpha=0.5)
        
        # Identity line
        max_rt = max(rt_max, y_obs.max(), y_pred.max())
        plt.plot([0, max_rt], [0, max_rt], color='red', linestyle='--', linewidth=1)
        
        stats_text = f"PCC = {cor:.4f}\nMAE = {mae:.2f} min\n$R^2$ = {r2:.4f}\n95% Range = {range_95:.2f} min\nN = {len(df)}"
        plt.text(0.05 * max_rt, 0.75 * max_rt, stats_text, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Observed RT (Minute)')
        plt.ylabel('Predicted RT (Minute)')
        plt.xlim(0, max_rt)
        plt.ylim(0, max_rt)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.title('RT Prediction Performance')
        
        output_plot = os.path.join(out_dir, "rt_performance_test.png")
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"RT performance plot saved to {output_plot}")
    except Exception as e:
        print(f"Error generating RT plot: {e}")


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
    parser.add_argument('--no_masking', action='store_true', help='disable masking for training')
    parser.add_argument('--user_mod', default=None, help='User defined modification')
    parser.add_argument('--verbose', type=int, default=1, help='log level. 1: info, 2: debug')

    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    out_prefix = args.out_prefix
    device = args.device.strip().lower()
    instrument = args.instrument
    nce = float(args.nce)
    tf_type = args.tf_type
    user_mod = args.user_mod

    if args.no_masking:
        use_valid = False
    else:
        use_valid = True

    # =========================
    # 2) CPU topology + affinity (before heavy imports)
    # =========================
    n_logical = os.cpu_count() or 1
    try:
        import psutil
        n_physical = psutil.cpu_count(logical=False) or n_logical
    except Exception:
        n_physical = max(1, n_logical // 2)

    configure_env_for_device(device, n_physical)

    import torch

    # Critical CPU tuning
    if device == "cpu":
        torch.set_num_threads(n_physical)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as e:
            print("[warn] set_num_interop_threads skipped:", e)

    if args.verbose >=2:
        print("torch intra", torch.get_num_threads())
        print("torch interop", torch.get_num_interop_threads())

    set_seed(int(args.seed))  # keep after torch import

    from peptdeep.settings import global_settings,add_user_defined_modifications
    import pandas as pd

    package_name = 'peptdeep'
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"The path of the {package_name} package is: {spec.origin}")
    else:
        print(f"The {package_name} package is not installed.")

    if user_mod is not None:
        user_mod = re.sub(r'^\"',"",user_mod)
        user_mod = re.sub(r'\"$',"",user_mod)
        umods = user_mod.split(";")
        mod_dict = {}
        for umod in umods:
            um = umod.split(",")
            mod_name = um[0]+"@"+um[1]
            mod_dict[mod_name] = {}
            mod_dict[mod_name]["composition"] = um[3]
            mod_dict[mod_name]["modloss_composition"] = ""

        print("User defined modifications:")
        print(mod_dict)
        add_user_defined_modifications(mod_dict)

    check_models_from_docker()

    if args.verbose >=2:
        print_thread_state(tag="Thread/BLAS summary (startup)")
    from threadpoolctl import threadpool_limits, threadpool_info
    if tf_type == "all" or tf_type == "test":
        if tf_type == "test":
            print("Test mode ...")
        model_mgr_rt = train_rt(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,device=device,threads=n_physical)
        plot_rt(out_dir)
        if os.path.exists(os.path.join(in_dir,"ccs_train_data.tsv")):
            ccs_data = pd.read_csv(os.path.join(in_dir,"ccs_train_data.tsv"),sep="\t")
            if ccs_data.shape[0] >= 100:
                model_mgr_ccs = train_ccs(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,device=device,threads=n_physical)
        model_mgr = train_ms2(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,log_transform=args.log_transform,nce=nce,device=device,instrument=instrument,use_valid=use_valid,threads=n_physical)
    elif tf_type == "rt":
        model_mgr_rt = train_rt(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,device=device,threads=n_physical)
        plot_rt(out_dir)
    elif tf_type == "ms2":
        model_mgr = train_ms2(in_dir=in_dir,out_dir=out_dir,mode_type=args.mode,log_transform=args.log_transform,nce=nce,device=device,instrument=instrument,use_valid=use_valid,threads=n_physical)

    try:
        from pyarrow import fs
        fs.finalize_s3()
    except:
        pass

    # Explicit GPU memory cleanup before forced exit
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all GPU ops to finish
        torch.cuda.empty_cache()  # Release cached memory back to driver

    os._exit(0)
