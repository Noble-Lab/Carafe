import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import importlib.util
import re
import json


def load_model_evaluation_metrics(model_dir: str) -> dict:
    """Load model evaluation metrics JSON from the model directory."""
    metrics_path = os.path.join(model_dir, "model_evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: Could not read {metrics_path}: {e}")
        return {}


def load_installed_model_by_mode(model_mgr, mode_type: str, model_name: str):
    """Load an installed pretrained model that matches the requested mode."""
    if mode_type == 'general':
        model_mgr.load_installed_models('generic', model_list=[model_name])
    elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
        model_mgr.load_installed_models('phos', model_list=[model_name])
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        model_mgr.load_installed_models('ubi', model_list=[model_name])
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        model_mgr.load_installed_models('generic', model_list=[model_name])


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
                verbose=1,
                torch_compile=False,
                threads=4):
    from models import ModelManager
    import alphabase.peptide.fragment as fragment
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
        model_mgr = ModelManager(mask_modloss=False, device=device)
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        model_mgr = ModelManager(mask_modloss=True, device=device)

    if model_dir == "generic":
        print("Load generic model!")
        load_installed_model_by_mode(model_mgr, mode_type, 'ms2')
    else:
        metrics = load_model_evaluation_metrics(model_dir)
        ms2_metrics = metrics.get("ms2", {}) if isinstance(metrics, dict) else {}
        use_finetuned_for_prediction = bool(ms2_metrics.get("use_finetuned_for_prediction", False))
        ms2_model_path = model_dir + "/ms2_model.pt"
        if use_finetuned_for_prediction and os.path.exists(ms2_model_path):
            print("Using fine-tuned MS2 model based on evaluation metrics")
            model_mgr.load_external_models(ms2_model_file=ms2_model_path)
        elif os.path.exists(ms2_model_path) and not metrics:
            print(f"Warning: {model_dir}/model_evaluation_metrics.json not found, loading generic MS2 model")
            load_installed_model_by_mode(model_mgr, mode_type, 'ms2')
        elif os.path.exists(ms2_model_path):
            print("Using pretrained MS2 model because fine-tuned metrics did not improve across all tracked metrics")
            load_installed_model_by_mode(model_mgr, mode_type, 'ms2')
        else:
            print(f"Warning: {model_dir}/ms2_model.pt not found, loading generic model")
            load_installed_model_by_mode(model_mgr, mode_type, 'ms2')

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
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=1, user_api="openmp"):
                # torch.set_num_threads(threads) # Already set in main
                if verbose >= 2:
                    print("torch intra", torch.get_num_threads())
                    print("torch interop", torch.get_num_interop_threads())
                pred_res = model_mgr.predict_ms2(a, torch_compile=torch_compile) 
    else:
        pred_res = model_mgr.predict_ms2(a, torch_compile=torch_compile)
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

    # Drop internal preprocessing columns that contain numpy arrays
    # These would corrupt TSV output due to numpy string formatting with newlines
    internal_cols = ['_aa_indices', '_mod_id_list', '_mod_site_list']
    output_df = a.drop(columns=[c for c in internal_cols if c in a.columns])
    
    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_df.parquet")
        output_df.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_df.tsv")
        output_df.to_csv(out_file, sep="\t", index=False)

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

    if mode_type == 'general' or mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mz_df = fragment.create_fragment_mz_dataframe(a['sequence,mods,mod_sites,charge,nAA'.split(',')],
                                                      ['b_z1','b_z2','y_z1','y_z2'], 
                                                      reference_fragment_df=None)
    elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
        mz_df = fragment.create_fragment_mz_dataframe(a['sequence,mods,mod_sites,charge,nAA'.split(',')],
                                                      ['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','b_modloss_z2','y_modloss_z1','y_modloss_z2'],
                                                      reference_fragment_df=None)

    
    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_mz_df.parquet")
        mz_df.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ms2_mz_df.tsv")
        mz_df.to_csv(out_file,sep="\t",index=False)

    import gc
    import torch
    del model_mgr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

def predict_rt(model_dir:str,
               pred_file:str,
               out_dir="./",
               out_prefix="rt_pred",
               device='gpu',
               mode_type='general',
               fast_mode=False,
               verbose=1,
               torch_compile=False,
               threads=4):
    from models import ModelManager
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
        model_mgr = ModelManager(mask_modloss=False, device=device)
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        model_mgr = ModelManager(mask_modloss=True, device=device)

    if model_dir == "generic":
        if mode_type == 'general':
            print("Load generic model!")
            model_mgr.load_installed_models('generic',  model_list=['rt'])
        elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
            print("Load generic model!")
            model_mgr.load_installed_models('phos', model_list=['rt'])
        elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
            print("Load generic model!")
            model_mgr.load_installed_models('ubi', model_list=['rt'])
        else:
            print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
            model_mgr.load_installed_models('generic', model_list=['rt'])
    else:
        if os.path.exists(model_dir+"/rt_model.pt"):
            model_mgr.load_external_models(rt_model_file=model_dir+"/rt_model.pt")
        else:
            print(f"Warning: {model_dir}/rt_model.pt not found, loading generic model")
            model_mgr.load_installed_models('generic', model_list=['rt'])

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
        # torch.set_num_threads(threads) # Already set in main
        if verbose >= 2:
            print("torch intra", torch.get_num_threads())
            print("torch interop", torch.get_num_interop_threads())
        pred_df = model_mgr.predict_rt(a, torch_compile=torch_compile)
    else:
        pred_df = model_mgr.predict_rt(a, torch_compile=torch_compile)
    pred_res = pred_df
    pred_res = model_mgr.rt_model.add_irt_column_to_precursor_df(pred_res)


    internal_cols = ['_aa_indices', '_mod_id_list', '_mod_site_list']
    output_df = pred_res.drop(columns=[c for c in internal_cols if c in pred_res.columns])

    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_rt_pred.parquet")
        output_df.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_rt_pred.tsv")
        output_df.to_csv(out_file,sep="\t",index=False)

    import gc
    import torch
    del model_mgr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_ccs(model_dir:str,
               pred_file:str,
               out_dir="./",
               out_prefix="ccs_pred",
               device='gpu',
               mode_type='general',
               fast_mode=False,
               verbose=1,
               torch_compile=False,
               threads=4):
    from models import ModelManager
    if mode_type == 'general':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
        model_mgr = ModelManager(mask_modloss=False, device=device)
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        model_mgr = ModelManager(mask_modloss=True, device=device)
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        model_mgr = ModelManager(mask_modloss=True, device=device)

    if model_dir == "generic":
        if mode_type == 'general':
            print("Load generic model!")
            model_mgr.load_installed_models('generic', model_list=['ccs'])
        elif mode_type == 'phosphorylation' or mode_type == 'phos' or mode_type == 'phospho':
            print("Load generic model!")
            model_mgr.load_installed_models('phos', model_list=['ccs'])
        elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
            print("Load generic model!")
            model_mgr.load_installed_models('ubi', model_list=['ccs'])
        else:
            print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
            model_mgr.load_installed_models('generic', model_list=['ccs'])
    else:
        if os.path.exists(model_dir+"/ccs_model.pt"):
            model_mgr.load_external_models(ccs_model_file=model_dir+"/ccs_model.pt")
        else:
            print(f"Warning: {model_dir}/ccs_model.pt not found, loading generic model")
            model_mgr.load_installed_models('generic', model_list=['ccs'])

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
        # torch.set_num_threads(threads) # Already set in main
        if verbose >= 2:
            print("torch intra", torch.get_num_threads())
            print("torch interop", torch.get_num_interop_threads())
        pred_df = model_mgr.predict_mobility(a, torch_compile=torch_compile)
    else:
        pred_df = model_mgr.predict_mobility(a, torch_compile=torch_compile)
    pred_res = pred_df
    #pred_res = model_mgr.rt_model.add_irt_column_to_precursor_df(pred_res)

    internal_cols = ['_aa_indices', '_mod_id_list', '_mod_site_list']
    output_df = pred_res.drop(columns=[c for c in internal_cols if c in pred_res.columns])

    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_ccs_pred.parquet")
        output_df.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_ccs_pred.tsv")
        output_df.to_csv(out_file,sep="\t",index=False)

    import gc
    import torch
    del model_mgr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# ENV setup BEFORE numpy/pandas/torch
# =========================
def configure_env_for_device(device: str, n_threads: int):
    """
    Must run BEFORE importing numpy/pandas/torch.
    """
    device = (device or "gpu").lower()

    if device == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_PROC_BIND", "true")

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
        from threadpoolctl import threadpool_info
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
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level')
    ## add log transform for intensity data
    parser.add_argument('--log_transform', action='store_true', help='log transform intensity data')
    parser.add_argument('--fast', action='store_true', help='Save data in parquet format to speed up reading and writing')
    parser.add_argument('--ccs', action='store_true', help='Predict CCS')
    parser.add_argument('--user_mod', default=None, help='User defined modification')
    parser.add_argument("--mod2mass", type=str, default=None, help="Change the mass of modifications, e.g., Deamidated@N=0")
    parser.add_argument("--torch_compile", action="store_true", help="Compile model with torch.compile for speed (CPU/GPU)")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads to use")
    parser.add_argument("--profile", action='store_true', help="Profile the script execution")
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

    if args.threads is None:
        args.threads = n_physical

    configure_env_for_device(device, args.threads)
    
    import torch
    import numpy as np
    # Prevent numpy from wrapping arrays with newlines when converting to strings
    # This fixes TSV file corruption when arrays are written as cell values
    np.set_printoptions(linewidth=np.inf)
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from models import ModelManager, global_settings, add_user_defined_modifications

    # Critical CPU tuning
    if device == "cpu":
        torch.set_num_threads(args.threads)
        if args.threads > 8:
            try:
                torch.set_num_interop_threads(2)
            except RuntimeError:
                pass

    if args.verbose >= 2:
        print("torch intra", torch.get_num_threads())
        print("torch interop", torch.get_num_interop_threads())



    set_seed(2024)
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


    def run_main():
        if args.verbose >= 2:
            print_thread_state(tag="Thread/BLAS summary (startup)")
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
                                   verbose=args.verbose,
                                   torch_compile=args.torch_compile,
                                   threads=args.threads)
            model_mgr = predict_rt(model_dir=args.model_dir, 
                               pred_file=args.in_file, 
                               out_dir=args.out_dir, 
                               out_prefix=args.out_prefix, 
                               device=args.device,
                               mode_type=args.mode,
                               fast_mode=args.fast,
                               verbose=args.verbose,
                               torch_compile=args.torch_compile,
                               threads=args.threads)
            if args.ccs:
                model_mgr_ccs = predict_ccs(model_dir=args.model_dir,
                                           pred_file=args.in_file,
                                           out_dir=args.out_dir,
                                           out_prefix=args.out_prefix,
                                           device=args.device,
                                           mode_type=args.mode,
                                           fast_mode=args.fast,
                                           verbose=args.verbose,
                                           torch_compile=args.torch_compile,
                                           threads=args.threads)
        elif args.tf_type == "rt":
            model_mgr = predict_rt(model_dir=args.model_dir, 
                               pred_file=args.in_file, 
                               out_dir=args.out_dir, 
                               out_prefix=args.out_prefix, 
                               device=args.device,
                               mode_type=args.mode,
                               fast_mode=args.fast,
                               verbose=args.verbose,
                               torch_compile=args.torch_compile,
                               threads=args.threads)
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
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
        elif args.tf_type == "ms2":
            model_mgr = predict_rt(model_dir="generic", 
                                    pred_file=args.in_file, 
                                    out_dir=args.out_dir, 
                                    out_prefix=args.out_prefix, 
                                    device=args.device,
                                    mode_type=args.mode,
                                    fast_mode=args.fast,
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
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
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
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
                                   verbose=args.verbose,
                                   torch_compile=args.torch_compile,
                                   threads=args.threads)
            model_mgr = predict_rt(args.model_dir, args.in_file, args.out_dir, args.out_prefix,
                    device=args.device, verbose=args.verbose, fast_mode=args.fast,
                    torch_compile=args.torch_compile, threads=args.threads)
            if args.ccs:
                model_mgr_ccs = predict_ccs(args.model_dir, args.in_file, args.out_dir, args.out_prefix,
                                    device=args.device,
                                    mode_type=args.mode,
                                    fast_mode=args.fast,
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile, threads=args.threads)
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
                                   verbose=args.verbose,
                                   torch_compile=args.torch_compile,
                                   threads=args.threads)
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
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
            if args.ccs:
                model_mgr_ccs = predict_ccs(model_dir="generic",
                                    pred_file=args.in_file,
                                    out_dir=pretrained_model_out_dir,
                                    out_prefix=args.out_prefix,
                                    device=args.device,
                                    mode_type=args.mode,
                                    fast_mode=args.fast,
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
        else:
            model_mgr = predict_rt(model_dir="generic", 
                                    pred_file=args.in_file, 
                                    out_dir=args.out_dir, 
                                    out_prefix=args.out_prefix, 
                                    device=args.device,
                                    mode_type=args.mode,
                                    fast_mode=args.fast,
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
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
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)
            if args.ccs:
                model_mgr_ccs = predict_ccs(model_dir="generic",
                                    pred_file=args.in_file,
                                    out_dir=args.out_dir,
                                    out_prefix=args.out_prefix,
                                    device=args.device,
                                    mode_type=args.mode,
                                    fast_mode=args.fast,
                                    verbose=args.verbose,
                                    torch_compile=args.torch_compile,
                                    threads=args.threads)

    if args.profile:
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()
        run_main()
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(30)
        print(s.getvalue())
        prof_file = os.path.join(args.out_dir, "prediction_profile.prof")
        pr.dump_stats(prof_file)
        print(f"Detailed profile saved to {prof_file}")
    else:
        run_main()

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
