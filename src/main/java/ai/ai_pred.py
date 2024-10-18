
from math import log, log10
from re import T
from sympy import true
import torch
import os
import numpy as np
import argparse
import sys
import alphabase.peptide.fragment as fragment
from peptdeep.settings import global_settings
import importlib.util
import sys

def predict_ms2(model_dir:str,
                pred_file:str,
                out_dir="./",
                out_prefix="ms2_pred",
                device='gpu',
                instrument='Eclipse',
                nce=27.0,
                mode_type='general',
                log_transform=False,
                fast_mode=False):
    import pandas as pd
    from peptdeep.pretrained_models import ModelManager
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
        a = pd.read_csv(pred_file,sep="\t",low_memory=False)
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    if device == 'cpu':
        ## get the number of cpus
        n_cpu = os.cpu_count()
        torch.set_num_threads(n_cpu)
    pred_res = model_mgr.predict_ms2(a) # the order of rows in a will be changed after the prediction if there is no nAA in it.
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

    if mode_type == 'general':
        mz_df = fragment.create_fragment_mz_dataframe(a['sequence,mods,mod_sites,charge'.split(',')], 
                                                      ['b_z1','b_z2','y_z1','y_z2'], 
                                                      reference_fragment_df=None)
    elif mode_type == 'phosphorylation':
        mz_df = fragment.create_fragment_mz_dataframe(a['sequence,mods,mod_sites,charge'.split(',')], 
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
               fast_mode=False):
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
        a = pd.read_csv(pred_file,sep="\t",low_memory=False)
    a.drop('charge', axis=1, inplace=True)
    a.drop_duplicates(inplace=True)
    a['mod_sites'] = a['mod_sites'].fillna("")
    a['mods'] = a['mods'].fillna("")
    if device == 'cpu':
        ## get the number of cpus
        n_cpu = os.cpu_count()
        torch.set_num_threads(n_cpu)
    pred_res = model_mgr.predict_rt(a)
    pred_res = model_mgr.rt_model.add_irt_column_to_precursor_df(pred_res)

    if fast_mode:
        out_file = os.path.join(out_dir,out_prefix+"_rt_pred.parquet")
        pred_res.to_parquet(out_file, compression='zstd')
    else:
        out_file = os.path.join(out_dir,out_prefix+"_rt_pred.tsv")
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


    args = parser.parse_args()

    if args.device == 'cpu':
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
                               fast_mode=args.fast)
        model_mgr = predict_rt(model_dir=args.model_dir, 
                           pred_file=args.in_file, 
                           out_dir=args.out_dir, 
                           out_prefix=args.out_prefix, 
                           device=args.device,
                           mode_type=args.mode,
                           fast_mode=args.fast)
    elif args.tf_type == "rt":
        model_mgr = predict_rt(model_dir=args.model_dir, 
                           pred_file=args.in_file, 
                           out_dir=args.out_dir, 
                           out_prefix=args.out_prefix, 
                           device=args.device,
                           mode_type=args.mode,
                           fast_mode=args.fast)
        model_mgr_rt = predict_ms2(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                fast_mode=args.fast)
    elif args.tf_type == "ms2":
        model_mgr = predict_rt(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast)
        model_mgr_rt = predict_ms2(model_dir=args.model_dir, 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                log_transform=args.log_transform,
                                fast_mode=args.fast)
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
                               fast_mode=args.fast)
        model_mgr = predict_rt(model_dir=args.model_dir, 
                           pred_file=args.in_file, 
                           out_dir=args.out_dir, 
                           out_prefix=args.out_prefix, 
                           device=args.device,
                           mode_type=args.mode,
                           fast_mode=args.fast)
        pretrained_model_out_dir = os.path.join(args.out_dir, "pretrained_models")
        if not os.path.exists(pretrained_model_out_dir):
            os.makedirs(pretrained_model_out_dir)
        model_mgr = predict_rt(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=pretrained_model_out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast)
        model_mgr_rt = predict_ms2(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=pretrained_model_out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                fast_mode=args.fast)
    else:
        model_mgr = predict_rt(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                mode_type=args.mode,
                                fast_mode=args.fast)
        model_mgr_rt = predict_ms2(model_dir="generic", 
                                pred_file=args.in_file, 
                                out_dir=args.out_dir, 
                                out_prefix=args.out_prefix, 
                                device=args.device,
                                instrument=args.instrument,
                                nce=float(args.nce),
                                mode_type=args.mode,
                                fast_mode=args.fast)
    
    

