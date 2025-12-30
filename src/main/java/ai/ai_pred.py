import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import importlib.util
import re


def predict_ms2(model_dir:str,
                pred_file:str,
                out_dir="./",
                out_prefix="ms2_pred",
                device='gpu',
                instrument='Eclipse',
                nce=27.0,
                mode_type='general',
                log_transform=False,
                fast_mode=False,
                mod2mass=None,
                verbose=1):
    import pandas as pd
    from peptdeep.pretrained_models import ModelManager
    import alphabase.peptide.fragment as fragment
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)

    if model_dir == "generic":
        print("Load generic model!")
        if mode_type == 'general':
            model_mgr.load_installed_models('generic')
        elif mode_type == 'phosphorylation':
            model_mgr.load_installed_models('phos')
    else:
        if os.path.exists(model_dir+"/ms2_model.pt"):
            model_mgr.load_external_models(ms2_model_file=model_dir+"/ms2_model.pt")
        else:
            print("Load generic model!")
            model_mgr.load_installed_models('generic')

    model_mgr.instrument = instrument
    model_mgr.nce = nce
    model_mgr.out_dir = out_dir
    model_mgr.verbose = False
    pd.options.mode.chained_assignment = None  # default=‘warn’
    if fast_mode:
        a = pd.read_parquet(pred_file)
    else:
        a = pd.read_csv(pred_file,sep="\t",low_memory=False,dtype={'mod_sites': str, 'mods': str})
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    if device == 'cpu':
        # CPU-only tuning: enforce pools during training
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=n_physical, user_api="openmp"):
                torch.set_num_threads(n_physical)
                if verbose >= 2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                pred_res = model_mgr.predict_ms2(a) # the order of rows in a will be changed after the prediction if there is no nAA in it.
    else:
        pred_res = model_mgr.predict_ms2(a)
    if mode_type == 'general':
        ## only use the following columns: 'b_z1','b_z2','y_z1','y_z2'
        pred_res = pred_res[['b_z1', 'b_z2', 'y_z1', 'y_z2']]

    if log_transform:
        print("log transform intensity data ...")
        # reverse the log transformation to pred_res
        pred_res = pred_res.apply(lambda x: np.power(10, x*np.log10(2))-1)

        

    
    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_pred.parquet")
        pred_res.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_pred.tsv")
        pred_res.to_csv(out_file,sep="\t",index=False)

    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_df.parquet")
        a.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_df.tsv")
        a.to_csv(out_file,sep="\t",index=False)

    if mod2mass is not None:
        from alphabase.constants.modification import MOD_MASS
        for mod in mod2mass.split(","):
            ## remove the first and last " when present
            if mod[0] == '"':
                mod = mod[1:]
            if mod[-1] == '"':
                mod = mod[:-1]
            mod = mod.split("=")
            print("Change the mass of modification: " + mod[0] + " to " + str(mod[1]))
            print("Before change:", MOD_MASS[mod[0]])
            MOD_MASS[mod[0]] = float(mod[1])
            print("After change:", MOD_MASS[mod[0]])

    if mode_type == 'general':
        mz_df = fragment.create_fragment_mz_dataframe(a['sequence,mods,mod_sites,charge,nAA'.split(',')],
                                                      ['b_z1','b_z2','y_z1','y_z2'], 
                                                      reference_fragment_df=None)
    elif mode_type == 'phosphorylation':
        mz_df = fragment.create_fragment_mz_dataframe(a['sequence,mods,mod_sites,charge,nAA'.split(',')],
                                                      ['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','b_modloss_z2','y_modloss_z1','y_modloss_z2'],
                                                      reference_fragment_df=None)

    
    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_mz_df.parquet")
        mz_df.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_mz_df.tsv")
        mz_df.to_csv(out_file,sep="\t",index=False)    
    

def predict_rt(model_dir:str,
               pred_file:str,
               out_dir="./",
               out_prefix="rt_pred",
               device='gpu',
               mode_type='general',
               fast_mode=False,
               verbose=1):
    import pandas as pd
    from peptdeep.pretrained_models import ModelManager
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)

    if model_dir == "generic":
        if mode_type == 'general':
            print("Load generic model!")
            model_mgr.load_installed_models('generic')
        elif mode_type == 'phosphorylation':
            print("Load generic model!")
            model_mgr.load_installed_models('phos')
    else:
        if os.path.exists(model_dir+"/rt_model.pt"):
            model_mgr.load_external_models(rt_model_file=model_dir+"/rt_model.pt")
        else:
            print("Load generic model!")
            model_mgr.load_installed_models('generic')

    # model_mgr.instrument = "Eclipse"
    model_mgr.out_dir = out_dir
    model_mgr.verbose = False
    pd.options.mode.chained_assignment = None  # default=‘warn’
    if fast_mode:
        a = pd.read_parquet(pred_file)
    else:
        a = pd.read_csv(pred_file,sep="\t",low_memory=False,dtype={'mod_sites': str, 'mods': str})
    a.drop('charge', axis=1, inplace=True)
    a.drop_duplicates(inplace=True)
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")

    if device == 'cpu':
        # CPU-only tuning: enforce pools during training
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=n_physical, user_api="openmp"):
                torch.set_num_threads(n_physical)
                if verbose >= 2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                pred_res = model_mgr.predict_rt(a)
    else:
        pred_res = model_mgr.predict_rt(a)
    pred_res = model_mgr.rt_model.add_irt_column_to_precursor_df(pred_res)

    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_rt_pred.parquet")
        pred_res.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_rt_pred.tsv")
        pred_res.to_csv(out_file,sep="\t",index=False)


def predict_ccs(model_dir:str,
               pred_file:str,
               out_dir="./",
               out_prefix="ccs_pred",
               device='gpu',
               mode_type='general',
               fast_mode=False,
               verbose=1):
    import pandas as pd
    from peptdeep.pretrained_models import ModelManager
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    elif mode_type == 'phosphorylation':
        model_mgr = ModelManager(mask_modloss=False, device=device)

    if model_dir == "generic":
        if mode_type == 'general':
            print("Load generic model!")
            model_mgr.load_installed_models('generic')
        elif mode_type == 'phosphorylation':
            print("Load generic model!")
            model_mgr.load_installed_models('phos')
    else:
        if os.path.exists(model_dir+"/ccs_model.pt"):
            model_mgr.load_external_models(ccs_model_file=model_dir+"/ccs_model.pt")
        else:
            print("Load generic model!")
            model_mgr.load_installed_models('generic')

    # model_mgr.instrument = "Eclipse"
    model_mgr.out_dir = out_dir
    model_mgr.verbose = False
    pd.options.mode.chained_assignment = None  # default=‘warn’
    if fast_mode:
        a = pd.read_parquet(pred_file)
    else:
        a = pd.read_csv(pred_file,sep="\t",low_memory=False,dtype={'mod_sites': str, 'mods': str})
    #a.drop('charge', axis=1, inplace=True)
    #a.drop_duplicates(inplace=True)
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    if device == 'cpu':
        # CPU-only tuning: enforce pools during training
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=n_physical, user_api="openmp"):
                torch.set_num_threads(n_physical)
                if verbose >= 2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                pred_res = model_mgr.predict_mobility(a)
    else:
        pred_res = model_mgr.predict_mobility(a)
    #pred_res = model_mgr.rt_model.add_irt_column_to_precursor_df(pred_res)

    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ccs_pred.parquet")
        pred_res.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ccs_pred.tsv")
        pred_res.to_csv(out_file,sep="\t",index=False)


def check_models_from_docker():
    model_zip = os.path.join(global_settings['PEPTDEEP_HOME'], "pretrained_models/pretrained_models.zip")
    if os.path.exists(model_zip):
        print("The pre-trained models:" + model_zip)
    elif os.path.exists("/data/peptdeep/pretrained_models/pretrained_models.zip"):
        global_settings['PEPTDEEP_HOME'] = "/data/peptdeep"
        print("The pre-trained models: /data/peptdeep/pretrained_models/pretrained_models.zip")
    else:
        print("Will download pre-trained models from github.")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    import sys
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict MS2 and RT.')
    parser.add_argument('--model_dir', default='generic', help='Model directory')
    parser.add_argument('--in_file', required=True, help='Input file')
    parser.add_argument('--out_dir', default='./', help='Output directory')
    parser.add_argument('--out_prefix', default="test", help='Output prefix')
    parser.add_argument('--device', default='gpu', help='Device to use for prediction')
    parser.add_argument('--instrument', default='Eclipse', help='For MS2 prediction')
    parser.add_argument('--nce', default=27, help='For MS2 prediction')
    parser.add_argument('--tf_type', default="all", help='For library generation: rt, ms2, or all')
    parser.add_argument('--mode', default="general", help='general or phosphorylation')
    ## add log transform for intensity data
    parser.add_argument('--log_transform', action='store_true', help='log transform intensity data')
    parser.add_argument('--fast', action='store_true', help='Save data in parquet format to speed up reading and writing')
    parser.add_argument('--ccs', action='store_true', help='Predict CCS')
    parser.add_argument('--mod2mass', default=None, help='Change the mass of modifications, e.g., Deamidated@N=0')
    parser.add_argument('--user_mod', default=None, help='User defined modification')
    parser.add_argument('--verbose', type=int, default=1, help='log level. 1: info, 2: debug')

    args = parser.parse_args()
    device = (args.device or "gpu").lower()
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
    import numpy as np
    # Critical CPU tuning
    if device == "cpu":
        torch.set_num_threads(n_physical)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as e:
            print("[warn] set_num_interop_threads skipped:", e)

    if args.verbose >= 2:
        print("torch intra", torch.get_num_threads())
        print("torch interop", torch.get_num_interop_threads())

    set_seed(2024)
    from peptdeep.settings import global_settings,add_user_defined_modifications
    package_name = 'peptdeep'
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"The path of the {package_name} package is: {spec.origin}")
    else:
        print(f"The {package_name} package is not installed.")

    check_models_from_docker()

    user_mod = args.user_mod
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

    import pandas as pd
    import numpy as np

    if args.verbose >= 2:
        print_thread_state(tag="Thread/BLAS summary (startup)")
    from threadpoolctl import threadpool_limits, threadpool_info
    if args.tf_type == "all":
        model_mgr_rt = predict_ms2(model_dir=args.model_dir, 
                               pred_file=args.in_file, 
                               out_dir=args.out_dir, 
                               out_prefix=args.out_prefix, 
                               device=args.device,
                               instrument=args.instrument,
                               nce=float(args.nce),
                               mode_type=args.mode,
                               log_transform=args.log_transform,
                               fast_mode=args.fast,
                               mod2mass=args.mod2mass,
                               verbose=args.verbose)
        model_mgr = predict_rt(model_dir=args.model_dir, 
                           pred_file=args.in_file, 
                           out_dir=args.out_dir, 
                           out_prefix=args.out_prefix, 
                           device=args.device,
                           mode_type=args.mode,
                           fast_mode=args.fast,
                           verbose=args.verbose)
        if args.ccs:
            model_mgr_ccs = predict_ccs(model_dir=args.model_dir,
                                       pred_file=args.in_file,
                                       out_dir=args.out_dir,
                                       out_prefix=args.out_prefix,
                                       device=args.device,
                                       mode_type=args.mode,
                                       fast_mode=args.fast,
                                       verbose=args.verbose)
    elif args.tf_type == "rt":
        model_mgr = predict_rt(model_dir=args.model_dir, 
                           pred_file=args.in_file, 
                           out_dir=args.out_dir, 
                           out_prefix=args.out_prefix, 
                           device=args.device,
                           mode_type=args.mode,
                           fast_mode=args.fast,
                           verbose=args.verbose)
        model_mgr_rt = predict_ms2(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                mod2mass=args.mod2mass,
                                verbose=args.verbose)
    elif args.tf_type == "ms2":
        model_mgr = predict_rt(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                verbose=args.verbose)
        model_mgr_rt = predict_ms2(model_dir=args.model_dir, 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                log_transform=args.log_transform,
                                fast_mode=args.fast,
                                mod2mass=args.mod2mass,
                                verbose=args.verbose)
    elif args.tf_type == "test":
        print("Test mode ...")
        model_mgr_rt = predict_ms2(model_dir=args.model_dir, 
                               pred_file=args.in_file, 
                               out_dir=args.out_dir, 
                               out_prefix=args.out_prefix, 
                               device=args.device,
                               instrument=args.instrument,
                               nce=float(args.nce),
                               mode_type=args.mode,
                               log_transform=args.log_transform,
                               fast_mode=args.fast,
                               mod2mass=args.mod2mass,
                               verbose=args.verbose)
        model_mgr = predict_rt(model_dir=args.model_dir, 
                           pred_file=args.in_file, 
                           out_dir=args.out_dir, 
                           out_prefix=args.out_prefix, 
                           device=args.device,
                           mode_type=args.mode,
                           fast_mode=args.fast,
                           verbose=args.verbose)
        if args.ccs:
            model_mgr_ccs = predict_ccs(model_dir=args.model_dir,
                                pred_file=args.in_file,
                                out_dir=args.out_dir,
                                out_prefix=args.out_prefix,
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                verbose=args.verbose)
        pretrained_model_out_dir = os.path.join(args.out_dir, "pretrained_models")
        if not os.path.exists(pretrained_model_out_dir):
            os.makedirs(pretrained_model_out_dir)
        model_mgr = predict_rt(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=pretrained_model_out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                verbose=args.verbose)
        model_mgr_rt = predict_ms2(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=pretrained_model_out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                mod2mass=args.mod2mass,
                                verbose=args.verbose)
        if args.ccs:
            model_mgr_ccs = predict_ccs(model_dir="generic",
                                pred_file=args.in_file,
                                out_dir=pretrained_model_out_dir,
                                out_prefix=args.out_prefix,
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                verbose=args.verbose)
    else:
        model_mgr = predict_rt(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                verbose=args.verbose)
        model_mgr_rt = predict_ms2(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                mod2mass=args.mod2mass,
                                verbose=args.verbose)
        if args.ccs:
            model_mgr_ccs = predict_ccs(model_dir="generic",
                                pred_file=args.in_file,
                                out_dir=args.out_dir,
                                out_prefix=args.out_prefix,
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast,
                                verbose=args.verbose)

    os._exit(0)
    
    

