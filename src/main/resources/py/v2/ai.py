import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import sys
import re
import logging
import math

# Set up peptdeep-style logging with timestamps
logging.basicConfig(
    format='%(asctime)s> %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

from alphabase.peptide.mobility import mobility_to_ccs_for_df

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
    
    import torch
    try:
        if torch.cuda.is_available():
            pass
        else:
            # Optimization: Flush denormals to zero to prevent performance drops on CPU
            # when gradients or weights become extremely small.
            if torch.set_flush_denormals(True):
                logging.info("Denormals flushed to zero for CPU performance.")
    except:
        pass


def print_thread_state(tag: str = ""):
    """Print thread/BLAS configuration for debugging"""
    import torch
    from threadpoolctl import threadpool_info
    if tag:
        print(f"\n==== {tag} ====")
    print("Python:", sys.version.split()[0])
    print("torch:", torch.__version__)
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


def set_seed(seed):
    """Set random seed for reproducibility"""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_steps_per_epoch(num_samples: int, batch_size: int) -> int:
    """Calculate steps per epoch."""
    if batch_size <= 0: return 0
    return math.ceil(num_samples / batch_size)


def adjust_batch_size(num_samples: int, current_batch_size: int, min_batch_size: int=32, min_steps_per_epoch: int=40) -> int:
    """Adjust batch size to ensure at least 40 steps per epoch."""
    steps = get_steps_per_epoch(num_samples, current_batch_size)
    if steps < min_steps_per_epoch:
        adjusted_batch_size = max(min_batch_size, math.ceil(num_samples / min_steps_per_epoch))
        # 2^floor(log2(adjusted_batch_size))
        adjusted_batch_size = 2 ** math.floor(math.log2(adjusted_batch_size))
        adjusted_batch_size = max(min_batch_size, adjusted_batch_size)
        logging.info(f"Adjusted batch size to {adjusted_batch_size} for at least {min_steps_per_epoch} steps/epoch.")
        return adjusted_batch_size
    return current_batch_size

def get_auto_tune_params(num_peptides: int, mode='ms2'):
    """
    Hybrid steps-based auto-tuning for consistent training effort.
    
    Instead of scaling epochs inversely with data size (which leads to 
    inconsistent gradient update counts), we fix a target training budget
    in steps and derive epochs accordingly.
    
    This ensures:
    - Consistent training effort (~2000 steps) across all dataset sizes.
    - Minimum epochs floor to prevent underfitting on small data.
    - Maximum epochs cap to prevent excessive training on tiny data.
    """
    if mode == 'ms2':
        target_steps = 2000  # Training budget in gradient updates
        batch_size = 512
        min_epochs = 15
        max_epochs = 50
        # LR: Start conservative, scale slightly with data size
        # Larger batches with more data benefit from slightly higher LR
        log_count = math.log10(max(1000, min(100000, num_peptides)))
        progress = (log_count - 3) / 2  # 0.0 at 1k, 1.0 at 100k
        lr = 1e-4 + (progress * 1e-4)  # 1e-4 â†’ 2e-4
    else:  # RT or CCS
        target_steps = 1000
        batch_size = 1024 # Conservative default for logic, but overridden below
        min_epochs = 20
        max_epochs = 60
        log_count = math.log10(max(1000, min(100000, num_peptides)))
        progress = (log_count - 3) / 2
        lr = 1e-4 + (progress * 1.5e-4) # 1e-4 â†’ 2.5e-4
    
    # Calculate epochs based on target budget
    # steps = (num_peptides / batch_size) * epochs
    # epochs = (steps * batch_size) / num_peptides
    if num_peptides > 0:
        calculated_epochs = int(math.ceil((target_steps * batch_size) / num_peptides))
    else:
        calculated_epochs = min_epochs
        
    final_epochs = max(min_epochs, min(max_epochs, calculated_epochs))
    
    return final_epochs, lr

def get_warmup_epochs(epochs):
    """Warmup: min 5, max 10, otherwise ~25% of epochs"""
    return min(10, max(5, epochs // 4))


def train_ms2(in_dir: str,
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
              use_best_model=False,
              auto_tune=False,
              early_stop=False,
              patience=10,
              verbose=1,
              epoch_to_train_ms2=20,
              warmup_epoch_to_train_ms2=10,
              batch_size_to_train_ms2=512,
              lr_to_train_ms2=0.0001,
              torch_compile=False,
              pretrained_model: str = None,
              adjust_batch_size_for_steps=True):
    """Train MS2 prediction model - replicates peptdeep exactly"""
    import torch
    from models import (ModelManager, psm_sampling_with_important_mods, 
                        normalize_fragment_intensities, calc_ms2_similarity_mask,
                        calc_ms2_similarity, settings)
    import numpy as np
    import pandas as pd
    import math
    
    pd.options.mode.chained_assignment = None
    a = pd.read_csv(in_dir + "/psm_pdv.txt", sep="\t", dtype={'mod_sites': str, 'mods': str})
    if "peptide" in a.columns:
        a["sequence"] = a["peptide"]
    a["nAA"] = a["sequence"].str.len()
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)

    # Calculate split sizes
    if mode_type == 'general':
        mask_modloss = True
    elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
        mask_modloss = False
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mask_modloss = True
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        mask_modloss = True
    model_mgr = ModelManager(mask_modloss=mask_modloss, device=device, out_dir=out_dir)
    
    if device == 'cpu':
        torch.set_num_threads(threads)
    # Load pretrained models
    try:
        if mode_type == 'general':
            model_mgr.load_installed_models('generic', model_list=['ms2'])
        elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
            model_mgr.load_installed_models('phos', model_list=['ms2'])
        elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
            model_mgr.load_installed_models('ubi', model_list=['ms2'])
        else:
            print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
            model_mgr.load_installed_models('generic', model_list=['ms2'])
    except Exception as e:
        print(f"Warning: Could not load pretrained models: {e}")

    # Load user-provided pretrained model if available
    if pretrained_model and os.path.exists(pretrained_model):
        try:
            print(f"Loading user-provided MS2 model from: {pretrained_model}")
            model_mgr.load_external_models(ms2_model_file=pretrained_model)
        except Exception as e:
            print(f"Error loading user-provided MS2 model: {e}")
    
    logging.info(f"MS2 Model Parameter Size: {model_mgr.ms2_model.get_parameter_num():,}")

    # Calculate split sizes
    n_test_ms2 = np.min([1000, int(math.ceil(a.shape[0] * 0.1) - 10)])
    psm_num_to_test_ms2 = n_test_ms2
    psm_num_to_train_ms2 = a.shape[0] - n_test_ms2
    
    # Training parameters
    if auto_tune:
        epoch_to_train_ms2, lr_to_train_ms2 = get_auto_tune_params(psm_num_to_train_ms2, mode='ms2')
        warmup_epoch_to_train_ms2 = get_warmup_epochs(epoch_to_train_ms2)
        logging.info(f"[Auto-Tune] Scaling MS2 params for {psm_num_to_train_ms2} peptides: epochs={epoch_to_train_ms2}, lr={lr_to_train_ms2:.1e}, batch={batch_size_to_train_ms2}")
    elif adjust_batch_size_for_steps:
        batch_size_to_train_ms2 = adjust_batch_size(psm_num_to_train_ms2, batch_size_to_train_ms2)
    
    psm_num_per_mod_to_train_ms2 = 50
    top_n_mods_to_train = 10
    
    # EXACT PARITY: PeptDeep v1 Sequence-Unique splitting
    # 1. Sample training data with important modifications (uses library default seed 1337)
    tr_df = psm_sampling_with_important_mods(
        a, psm_num_to_train_ms2, top_n_mods_to_train, psm_num_per_mod_to_train_ms2
    ).copy()
    
    # 2. Split test set by excluding training sequences (peptdeep default)
    test_psm_df = a[~a.sequence.isin(set(tr_df.sequence))].copy()
    
    # 3. Sample test set to requested size using hardcoded seed 1337
    if len(test_psm_df) > psm_num_to_test_ms2:
        test_psm_df = test_psm_df.sample(psm_num_to_test_ms2, random_state=1337).copy()
    elif len(test_psm_df) == 0:
        test_psm_df = a.copy()
    
    print(f"The number of PSMs to train MS2 model: {len(tr_df)}")
    print(f"The number of PSMs to test MS2 model: {len(test_psm_df)}")
    steps = get_steps_per_epoch(len(tr_df), batch_size_to_train_ms2)
    print(f"Training steps per epoch: {steps}")
    
    model_mgr.nce = nce
    model_mgr.instrument = instrument

    if verbose >=2:
        print("Input PSM dtypes:")
        print(a.dtypes)
    
    b = pd.read_csv(in_dir + "/fragment_intensity_df.tsv", sep="\t", engine="pyarrow")
    
    if log_transform:
        print("log transform intensity data ...")
        b = b.apply(lambda x: np.log10(x + 1) / np.log10(2))
    
    valid = None
    if use_valid and os.path.exists(in_dir + "/fragment_intensity_valid.tsv"):
        print("Use valid intensity data ...")
        valid = pd.read_csv(in_dir + "/fragment_intensity_valid.tsv", sep="\t", engine="pyarrow")
    
    # Prepare training intensity dataframes
    tr_inten_df = pd.DataFrame()
    if valid is not None:
        tr_inten_valid_df = pd.DataFrame()
    else:
        tr_inten_valid_df = None

    for frag_type in model_mgr.ms2_model.charged_frag_types:
        if frag_type in b.columns:
            tr_inten_df[frag_type] = b[frag_type]
        else:
            tr_inten_df[frag_type] = 0.0
        
        if tr_inten_valid_df is not None:
            if frag_type in valid.columns:
                tr_inten_valid_df[frag_type] = valid[frag_type]
            else:
                tr_inten_valid_df[frag_type] = 0.0
    
    # Normalize intensities (like peptdeep)
    normalize_fragment_intensities(tr_df, tr_inten_df)
    
    # Advanced DIA Enhancements: Auto-Tune Scaling
    if auto_tune:
        logging.info("[Auto-Tune] Enhanced DIA optimization enabled (Auto-Scaling).")

    # Set NCE and instrument
    model_mgr_settings = settings['model_mgr']
    # use_grid_nce_search = model_mgr_settings['transfer']['grid_nce_search'] # Logic from ai.py uses arg
    use_grid_nce_search = args.use_grid_nce_search
    
    if use_grid_nce_search:
        print("Perform grid search for NCE and instrument ...")
        nce, instrument = model_mgr.ms2_model.grid_nce_search(
            tr_df.copy(), tr_inten_df,
            nce_first=model_mgr_settings['transfer']['grid_nce_first'],
            nce_last=model_mgr_settings['transfer']['grid_nce_last'],
            nce_step=model_mgr_settings['transfer']['grid_nce_step'],
            search_instruments=model_mgr_settings['transfer']['grid_instrument'],
        )
        tr_df['nce'] = int(nce)
        tr_df['instrument'] = instrument
        print(f"Optimal NCE: {nce}, Instrument: {instrument}")
    else:
        if 'nce' not in tr_df.columns:
            tr_df['nce'] = nce
        if 'instrument' not in tr_df.columns:
            tr_df['instrument'] = model_mgr.instrument
    
    # Prepare test set NCE/Instrument
    if 'nce' not in test_psm_df.columns:
        test_psm_df['nce'] = nce
    if 'instrument' not in test_psm_df.columns:
        test_psm_df['instrument'] = model_mgr.instrument
    
    # Pre-training evaluation
    pretrained_metrics = None
    if len(test_psm_df) > 0:
        print("Using masking in metrics calculation ...")
        pred_frag_df = model_mgr.ms2_model.predict(test_psm_df.copy(), reference_frag_df=tr_inten_df)
        test_res = calc_ms2_similarity_mask(
            test_psm_df.copy(), pred_frag_df, tr_inten_df, tr_inten_valid_df
        )
        out_test_res_to_file = os.path.join(out_dir, 'test_res_pretrained.csv')
        test_res[0].to_csv(out_test_res_to_file, index=False)
        # Store pretrained metrics for comparison
        pretrained_metrics = {
            'PCC': test_res[0]['PCC'].median(),
            'COS': test_res[0]['COS'].median(),
            'SA': test_res[0]['SA'].median(),
            'SPC': test_res[0]['SPC'].median()
        }
        logging.info("Testing pretrained MS2 model:\n" + str(test_res[-1]))
    
    # Training with per-epoch test evaluation
    if len(tr_df) > 0:
        logging.info(f"{len(tr_df)} PSMs for MS2 model training/transfer learning")
        logging.info(f"Training with fixed sequence length: 0")
        
        import torch
        # Prepare for training
        ms2_model = model_mgr.ms2_model
        if torch_compile:
            ms2_model.compile_model()
        ms2_model._prepare_training(tr_df, lr_to_train_ms2, 
                                    fragment_intensity_df=tr_inten_df,
                                    fragment_intensity_df_valid=tr_inten_valid_df)
        
        # EXACT PARITY: Re-enabling LR warmup (10 epochs) as in original script
        lr_scheduler = ms2_model._get_lr_schedule_with_warmup(warmup_epoch_to_train_ms2, epoch_to_train_ms2)
        
        best_test_loss = [float('inf')]
        epochs_without_improvement = [0]
        stopped_early = [False]

        def run_training_loop():
            for ep in range(epoch_to_train_ms2):
                # Train one epoch
                batch_cost = ms2_model._train_one_epoch(
                    tr_df, ep, batch_size_to_train_ms2, False,
                    fragment_intensity_df=tr_inten_df,
                    fragment_intensity_df_valid=tr_inten_valid_df
                )
                lr_scheduler.step()
                train_loss = np.mean(batch_cost)
                
                # Evaluate on test data after each epoch
                test_loss_str = ""
                test_loss = 0
                test_losses = []
                if len(test_psm_df) > 0:
                    # Calculate masked test loss (L1 loss consistent with training)
                    ms2_model.model.eval()
                    with torch.no_grad():
                        for nAA, df_group in test_psm_df.groupby('nAA', sort=False):
                            for i in range(0, len(df_group), batch_size_to_train_ms2):
                                batch_df = df_group.iloc[i:i + batch_size_to_train_ms2, :]
                                targets = ms2_model._get_targets_from_batch_df(
                                    batch_df, fragment_intensity_df=tr_inten_df
                                )
                                features = ms2_model._get_features_from_batch_df(batch_df)
                                predicts = ms2_model.model(*features)
                                
                                if tr_inten_valid_df is not None:
                                    # Apply masking consistent with training
                                    valid_targets = ms2_model._get_valid_targets_from_batch_df(
                                        batch_df, fragment_intensity_df_valid=tr_inten_valid_df
                                    )
                                    mask = torch.where(valid_targets <= 0, 1.0, 0.0)
                                    num_valid = torch.sum(mask)
                                    if num_valid > 0:
                                        loss = torch.nn.L1Loss(reduction='sum')(mask * predicts, mask * targets) / num_valid
                                        test_losses.append(loss.item())
                                else:
                                    loss = torch.nn.L1Loss()(predicts, targets)
                                    test_losses.append(loss.item())
                    ms2_model.model.train()
                    
                    if test_losses:
                        test_loss = np.mean(test_losses)
                        test_loss_str = f", test_loss={test_loss}"
                
                ms2_model.history['epoch'].append(ep + 1)
                ms2_model.history['lr'].append(lr_scheduler.get_last_lr()[0])
                ms2_model.history['train_loss'].append(train_loss)
                ms2_model.history['test_loss'].append(test_loss if test_losses else 0)

                # Checkpoint best model and track improvement
                improved = False
                if test_losses:
                    if test_loss < best_test_loss[0]:
                        best_test_loss[0] = test_loss
                        improved = True
                        epochs_without_improvement[0] = 0
                        if use_best_model:
                            ms2_model.save(out_dir + "/ms2_model.pt")
                            if verbose:
                                logging.info(f"New best MS2 model saved at epoch {ep + 1} with test_loss={test_loss:.4f}")
                    else:
                        epochs_without_improvement[0] += 1
                
                mask_str = " (Masked)" if tr_inten_valid_df is not None else ""
                logging.info(f"[Training] Epoch={ep + 1}{mask_str}, lr={lr_scheduler.get_last_lr()[0]:.1e}, train_loss={train_loss}{test_loss_str}")
                
                # Early stopping check
                if early_stop and epochs_without_improvement[0] >= patience:
                    logging.info(f"Early stopping triggered at epoch {ep + 1} (no improvement for {patience} epochs)")
                    stopped_early[0] = True
                    break
            
            torch.cuda.empty_cache()
        
        if device == 'cpu':
            from threadpoolctl import threadpool_limits
            # Allow BLAS to use the threads as well for matrix math performance
            with threadpool_limits(limits=threads, user_api="blas"):
                with threadpool_limits(limits=threads, user_api="openmp"):
                    if device == 'cpu':
                        torch.set_num_threads(threads)
                    run_training_loop()
        else:
            run_training_loop()
    
    # Pre-reload evaluation
    test_psm_df_clean = None
    if len(test_psm_df) > 0 and use_best_model:
        test_psm_df_clean = test_psm_df.copy()
        # Ensure indices remain integer to prevent IndexError in predict/normalize
        for col in ['frag_start_idx', 'frag_stop_idx', 'psm_id', 'ms2_scan']:
            if col in test_psm_df.columns:
                test_psm_df[col] = test_psm_df[col].astype(np.int64)

        pred_frag_df = model_mgr.ms2_model.predict(test_psm_df, reference_frag_df=tr_inten_df)
        test_res = calc_ms2_similarity_mask(
            test_psm_df, pred_frag_df, tr_inten_df, tr_inten_valid_df
        )
        logging.info("Performance of the model at the end of training (before reload):\n" + str(test_res[-1]))

    # Reload best model if saved and requested
    if use_best_model and os.path.exists(out_dir + "/ms2_model.pt"):
        model_mgr.ms2_model.load(out_dir + "/ms2_model.pt")
        logging.info("Reloaded best MS2 model from checkpoint.")

    # Post-training evaluation
    if len(test_psm_df) > 0:
        if test_psm_df_clean is not None:
            test_psm_df = test_psm_df_clean
        # verbose: 1=normal, 2=debug
        if verbose >=2:
            print("test_psm_df dtypes:")
            print(test_psm_df.dtypes)
        
        print("Using masking in metrics calculation ...")
        
        # Ensure indices remain integer to prevent IndexError in predict/normalize
        for col in ['frag_start_idx', 'frag_stop_idx', 'psm_id', 'ms2_scan']:
            if col in test_psm_df.columns:
                test_psm_df[col] = test_psm_df[col].astype(np.int64)

        pred_frag_df = model_mgr.ms2_model.predict(test_psm_df, reference_frag_df=tr_inten_df)
        test_res = calc_ms2_similarity_mask(
            test_psm_df, pred_frag_df, tr_inten_df, tr_inten_valid_df
        )
        out_test_res_to_file = os.path.join(out_dir, 'test_res_fine_tuned.csv')
        test_res[0].to_csv(out_test_res_to_file, index=False)
        
        eval_label = "Final Evaluation (Best Model)" if use_best_model else "Final Evaluation (Last Model)"
        logging.info(f"{eval_label}:\n" + str(test_res[-1]))
        
        # Print improvement summary comparing pretrained vs fine-tuned
        if pretrained_metrics is not None:
            finetuned_metrics = {
                'PCC': test_res[0]['PCC'].median(),
                'COS': test_res[0]['COS'].median(),
                'SA': test_res[0]['SA'].median(),
                'SPC': test_res[0]['SPC'].median()
            }
            logging.info("\n=== Improvement Summary (Median) ===")
            logging.info(f"{'Metric':<6} {'Pretrained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
            logging.info("-" * 44)
            for metric in ['PCC', 'COS', 'SA', 'SPC']:
                pre = pretrained_metrics[metric]
                post = finetuned_metrics[metric]
                delta = post - pre
                sign = "+" if delta >= 0 else ""
                logging.info(f"{metric:<6} {pre:>12.4f} {post:>12.4f} {sign}{delta:>11.4f}")
    
    model_mgr.ms2_model.save(out_dir + "/ms2_model.pt")
    model_mgr.ms2_model.save_training_history(os.path.join(out_dir, "ms2_history.tsv"))
    model_mgr.ms2_model.plot_training_history(os.path.join(out_dir, "ms2_history.png"), mark_best=use_best_model)
    return model_mgr



def train_rt(in_dir: str, out_dir: str, mode_type="general", device='gpu', threads=1, use_best_model=False, auto_tune=False, early_stop=False, patience=10, verbose=1,
             epoch_to_train_rt_ccs=40, warmup_epoch_to_train_rt_ccs=10, batch_size_to_train_rt_ccs=1024, lr_to_train_rt_ccs=0.0001, torch_compile=False,
             pretrained_model: str = None, adjust_batch_size_for_steps=True):
    """Train RT prediction model - replicates peptdeep exactly"""
    import pandas as pd
    import numpy as np
    import math
    from models import (ModelManager, psm_sampling_with_important_mods, 
                        evaluate_linear_regression)
    
    import torch
    pd.options.mode.chained_assignment = None
    a = pd.read_csv(in_dir + "/rt_train_data.tsv", sep="\t", dtype={'mod_sites': str, 'mods': str})
    
    if "sequence" not in a.columns and "peptide" in a.columns:
        a["sequence"] = a["peptide"]
    
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)
    
    # Initialize ModelManager
    if mode_type == 'general':
        mask_modloss = True
    elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
        mask_modloss = False
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mask_modloss = True
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        mask_modloss = True
    model_mgr = ModelManager(mask_modloss=mask_modloss, device=device, out_dir=out_dir)
    
    if device == 'cpu':
        torch.set_num_threads(threads)
    # Load pretrained models
    try:
        if mode_type == 'general':
            model_mgr.load_installed_models('generic', model_list=['rt'])
        elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
            model_mgr.load_installed_models('phos', model_list=['rt'])
        elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
            model_mgr.load_installed_models('ubi', model_list=['rt'])
        else:
            print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
            model_mgr.load_installed_models('generic', model_list=['rt'])
    except Exception as e:
        print(f"Warning: Could not load pretrained models: {e}")

    # Load user-provided pretrained model if available
    if pretrained_model and os.path.exists(pretrained_model):
        try:
            print(f"Loading user-provided RT model from: {pretrained_model}")
            model_mgr.load_external_models(rt_model_file=pretrained_model)
        except Exception as e:
            print(f"Error loading user-provided RT model: {e}")

    logging.info(f"RT Model Parameter Size: {model_mgr.rt_model.get_parameter_num():,}")

    psm_num_per_mod_to_train_rt_ccs = 50
    top_n_mods_to_train = 10

    n_test_rt = np.min([1000, int(math.ceil(a.shape[0] * 0.1) - 10)])
    psm_num_to_test_rt_ccs = n_test_rt
    psm_num_to_train_rt_ccs = a.shape[0] - n_test_rt

    # Training parameters
    if auto_tune:
        epoch_to_train_rt_ccs, lr_to_train_rt_ccs = get_auto_tune_params(psm_num_to_train_rt_ccs, mode='rt_ccs')
        warmup_epoch_to_train_rt_ccs = get_warmup_epochs(epoch_to_train_rt_ccs)
        logging.info(f"[Auto-Tune] Scaling RT params for {psm_num_to_train_rt_ccs} peptides: epochs={epoch_to_train_rt_ccs}, lr={lr_to_train_rt_ccs:.1e}, batch={batch_size_to_train_rt_ccs}")
    elif adjust_batch_size_for_steps:
        batch_size_to_train_rt_ccs = adjust_batch_size(psm_num_to_train_rt_ccs, batch_size_to_train_rt_ccs)
        

    print(f"The number of peptides to train RT model: {psm_num_to_train_rt_ccs}")
    print(f"The number of peptides to test RT model: {psm_num_to_test_rt_ccs}")

    # EXACT PARITY: PeptDeep v1 internal data preparation
    # 1. Median Grouping
    psm_df = a.groupby(['sequence','mods','mod_sites'])[['rt_norm']].median().reset_index(drop=False)
    
    # 2. Sample (Seed 1337 default)
    tr_df = psm_sampling_with_important_mods(
        psm_df, psm_num_to_train_rt_ccs, top_n_mods_to_train, psm_num_per_mod_to_train_rt_ccs
    ).copy()
    
    # Add nAA metadata after sampling
    psm_df['nAA'] = psm_df['sequence'].str.len()
    tr_df['nAA'] = tr_df['sequence'].str.len()
    
    # 3. Split test (Sequence-Unique)
    test_psm_df = psm_df[~psm_df.sequence.isin(set(tr_df.sequence))].copy()
    
    # 4. Sample test (Seed 1337)
    if len(test_psm_df) > psm_num_to_test_rt_ccs:
        test_psm_df = test_psm_df.sample(psm_num_to_test_rt_ccs, random_state=1337).copy()
    elif len(test_psm_df) == 0:
        test_psm_df = psm_df.copy()
    
    logging.info(f"{len(tr_df)} PSMs for RT model training/transfer learning")
    steps = get_steps_per_epoch(len(tr_df), batch_size_to_train_rt_ccs)
    logging.info(f"Training steps per epoch: {steps}")
    
    rt_model = model_mgr.rt_model
    if torch_compile:
        rt_model.compile_model()

    # Pre-training evaluation
    pretrained_metrics = None
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import median_absolute_error,r2_score
            test_psm_df['rt_pred'] = rt_model.predict(test_psm_df.copy())['rt_pred']
            y_obs = test_psm_df['rt_norm'].values
            y_pred = test_psm_df['rt_pred'].values
            pre_r = scipy.stats.pearsonr(y_obs, y_pred)[0]
            pre_mae = median_absolute_error(y_obs, y_pred)
            r2 = r2_score(y_obs, y_pred)
            pretrained_metrics = {'R2': r2, 'MAE (normalized)': pre_mae}
            logging.info(f"Testing pretrained RT model: R2={r2:.4f}, MAE (normalized)={pre_mae:.4f}")
            
            # Save pretrained results and generate plot
            test_psm_df.to_csv(os.path.join(out_dir, "rt_test_pretrained.tsv"), sep="\t", index=False)
            plot_rt(in_dir, out_dir, test_file="rt_test_pretrained.tsv", 
                    output_plot="rt_performance_pretrained.png", 
                    title="RT Prediction Performance (Pretrained)",
                    denormalize_rt=False)
        except Exception as e:
            logging.warning(f"Could not perform pre-training RT evaluation: {e}")
    
    rt_model._prepare_training(tr_df, lr_to_train_rt_ccs)
    lr_scheduler = rt_model._get_lr_schedule_with_warmup(warmup_epoch_to_train_rt_ccs, epoch_to_train_rt_ccs)
    
    best_test_loss = [float('inf')]
    epochs_without_improvement = [0]
    stopped_early = [False]

    def run_training_loop():
        for ep in range(epoch_to_train_rt_ccs):
            batch_cost = rt_model._train_one_epoch(tr_df, ep, batch_size_to_train_rt_ccs, False)
            lr_scheduler.step()
            train_loss = np.mean(batch_cost)
            
            test_loss_str = ""
            test_loss = 0
            test_losses = []
            if test_psm_df is not None and len(test_psm_df) > 0:
                rt_model.model.eval()
                with torch.no_grad():
                    for nAA, df_group in test_psm_df.groupby('nAA', sort=False):
                        for i in range(0, len(df_group), batch_size_to_train_rt_ccs):
                            batch_df = df_group.iloc[i:i + batch_size_to_train_rt_ccs, :]
                            targets = rt_model._get_targets_from_batch_df(batch_df)
                            features = rt_model._get_features_from_batch_df(batch_df)
                            predicts = rt_model.model(*features)
                            loss = torch.nn.L1Loss()(predicts, targets)
                            test_losses.append(loss.item())
                rt_model.model.train()
                if test_losses:
                    test_loss = np.mean(test_losses)
                    test_loss_str = f", test_loss={test_loss}"
            
            rt_model.history['epoch'].append(ep + 1)
            rt_model.history['lr'].append(lr_scheduler.get_last_lr()[0])
            rt_model.history['train_loss'].append(train_loss)
            rt_model.history['test_loss'].append(test_loss if test_losses else 0)

            if test_losses:
                if test_loss < best_test_loss[0]:
                    best_test_loss[0] = test_loss
                    epochs_without_improvement[0] = 0
                    if use_best_model:
                        rt_model.save(out_dir + "/rt_model.pt")
                else:
                    epochs_without_improvement[0] += 1
            
            logging.info(f"[Training] Epoch={ep + 1}, lr={lr_scheduler.get_last_lr()[0]:.1e}, train_loss={train_loss}{test_loss_str}")
            
            if early_stop and epochs_without_improvement[0] >= patience:
                stopped_early[0] = True
                break
        torch.cuda.empty_cache()

    run_training_loop()
    if use_best_model and os.path.exists(out_dir + "/rt_model.pt"):
        rt_model.load(out_dir + "/rt_model.pt")
    
    rt_model.save(out_dir + "/rt_model.pt")

    # Post-training evaluation
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import median_absolute_error,r2_score
            test_psm_df['rt_pred'] = rt_model.predict(test_psm_df.copy())['rt_pred']
            y_obs = test_psm_df['rt_norm'].values
            y_pred = test_psm_df['rt_pred'].values
            post_r = scipy.stats.pearsonr(y_obs, y_pred)[0]
            post_mae = median_absolute_error(y_obs, y_pred)
            post_r2 = r2_score(y_obs, y_pred)
            
            eval_label = "Final Evaluation (Best Model)" if use_best_model else "Final Evaluation (Last Model)"
            logging.info(f"{eval_label}: R2={post_r2:.4f}, MAE (normalized)={post_mae:.4f}")
            
            if pretrained_metrics is not None:
                logging.info("\n=== Improvement Summary (RT) ===")
                logging.info(f"{'Metric':<20} {'Pretrained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
                logging.info("-" * 60)
                for metric in ['R2', 'MAE (normalized)']:
                    pre = pretrained_metrics[metric]
                    if metric.lower().startswith("r2"):
                        post = post_r2
                    elif metric.lower().startswith("mae"):
                        post = post_mae
                    else:
                        print(f"Unknown metric: {metric}")
                        continue
                    delta = post - pre
                    if metric.lower().startswith("r2"):
                        sign = "+" if delta >= 0 else "-"
                    elif metric.lower().startswith("mae"):
                        sign = "-" if delta >= 0 else "+"
                    else:
                        print(f"Unknown metric: {metric}")
                        continue
                    logging.info(f"{metric:<20} {pre:>12.4f} {post:>12.4f} {sign}{delta:>11.4f}")
            
            test_psm_df.to_csv(os.path.join(out_dir, "rt_test.tsv"), sep="\t", index=False)
        except Exception as e:
            logging.warning(f"Could not perform post-training RT evaluation: {e}")

    rt_model.save_training_history(os.path.join(out_dir, "rt_history.tsv"))
    rt_model.plot_training_history(os.path.join(out_dir, "rt_history.png"), mark_best=use_best_model)
    return model_mgr

def train_ccs(in_dir: str, out_dir: str, mode_type="general", device='gpu', threads=1, use_best_model=False, auto_tune=False, early_stop=False, patience=10, verbose=1,
              epoch_to_train_rt_ccs=40, warmup_epoch_to_train_rt_ccs=10, batch_size_to_train_rt_ccs=1024, lr_to_train_rt_ccs=0.0001, torch_compile=False,
              pretrained_model: str = None,
              adjust_batch_size_for_steps=True):
    """Train CCS prediction model - replicates peptdeep exactly"""
    import pandas as pd
    import numpy as np
    import math
    from models import (ModelManager, psm_sampling_with_important_mods,
                        evaluate_linear_regression)
    import torch
    pd.options.mode.chained_assignment = None
    a = pd.read_csv(in_dir + "/ccs_train_data.tsv", sep="\t", dtype={'mod_sites': str, 'mods': str})
    
    if "sequence" not in a.columns and "peptide" in a.columns:
        a["sequence"] = a["peptide"]
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)
    
    if mode_type == 'general':
        mask_modloss = True
    elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
        mask_modloss = False
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mask_modloss = True
    else:
        mask_modloss = True
    model_mgr = ModelManager(mask_modloss=mask_modloss, device=device, out_dir=out_dir)
    
    if device == 'cpu':
        torch.set_num_threads(threads)
    try:
        if mode_type == 'general':
            model_mgr.load_installed_models('generic', model_list=['ccs'])
        elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
            model_mgr.load_installed_models('phos', model_list=['ccs'])
        elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
            model_mgr.load_installed_models('ubi', model_list=['ccs'])
        else:
            model_mgr.load_installed_models('generic', model_list=['ccs'])
    except Exception as e:
        print(f"Warning: Could not load pretrained models: {e}")

    # Load user-provided pretrained model if available
    if pretrained_model and os.path.exists(pretrained_model):
        try:
            print(f"Loading user-provided CCS model from: {pretrained_model}")
            model_mgr.load_external_models(ccs_model_file=pretrained_model)
        except Exception as e:
            print(f"Error loading user-provided CCS model: {e}")

    logging.info(f"CCS Model Parameter Size: {model_mgr.ccs_model.get_parameter_num():,}")

    n_test_ccs = np.min([1000, int(math.ceil(a.shape[0] * 0.1) - 10)])
    psm_num_to_test_rt_ccs = n_test_ccs
    psm_num_to_train_rt_ccs = a.shape[0] - n_test_ccs

    if auto_tune:
        epoch_to_train_rt_ccs, lr_to_train_rt_ccs = get_auto_tune_params(psm_num_to_train_rt_ccs, mode='rt_ccs')
        warmup_epoch_to_train_rt_ccs = get_warmup_epochs(epoch_to_train_rt_ccs)
    elif adjust_batch_size_for_steps:
        batch_size_to_train_rt_ccs = adjust_batch_size(psm_num_to_train_rt_ccs, batch_size_to_train_rt_ccs)

    print(f"The number of peptides to train CCS model: {psm_num_to_train_rt_ccs}")
    print(f"The number of peptides to test CCS model: {psm_num_to_test_rt_ccs}")

    if 'ccs' not in a.columns and 'mobility' in a.columns:
        a['ccs'] = mobility_to_ccs_for_df(a, 'mobility')
    
    # 1. Median Grouping
    psm_df = a.groupby(['sequence','mods','mod_sites','charge'])[['mobility','ccs']].median().reset_index(drop=False)
    
    # 2. Sample
    tr_df = psm_sampling_with_important_mods(
        psm_df, psm_num_to_train_rt_ccs, 10, 50
    ).copy()
    
    # Add nAA metadata after sampling
    psm_df['nAA'] = psm_df['sequence'].str.len()
    tr_df['nAA'] = tr_df['sequence'].str.len()
    
    test_psm_df = psm_df[~psm_df.sequence.isin(set(tr_df.sequence))].copy()
    if len(test_psm_df) > psm_num_to_test_rt_ccs:
        test_psm_df = test_psm_df.sample(psm_num_to_test_rt_ccs, random_state=1337).copy()
    elif len(test_psm_df) == 0:
        test_psm_df = psm_df.copy()
    
    logging.info(f"{len(tr_df)} PSMs for CCS model training/transfer learning")
    steps = get_steps_per_epoch(len(tr_df), batch_size_to_train_rt_ccs)
    logging.info(f"Training steps per epoch: {steps}")

    ccs_model = model_mgr.ccs_model
    if torch_compile:
        ccs_model.compile_model()

    # Pre-training evaluation
    pretrained_metrics = None
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import median_absolute_error,r2_score
            test_psm_df['ccs_pred'] = ccs_model.predict(test_psm_df.copy())['ccs_pred']
            # Also need mobility_pred for plotting
            test_psm_df['mobility_pred'] = ccs_model.ccs_to_mobility_pred(test_psm_df)
            
            y_obs_ccs = test_psm_df['ccs'].values
            y_pred_ccs = test_psm_df['ccs_pred'].values
            pre_r_ccs = scipy.stats.pearsonr(y_obs_ccs, y_pred_ccs)[0]
            pre_mae_ccs = median_absolute_error(y_obs_ccs, y_pred_ccs)
            pre_r2_ccs = r2_score(y_obs_ccs, y_pred_ccs)
            
            y_obs_mobility = test_psm_df['mobility'].values
            y_pred_mobility = test_psm_df['mobility_pred'].values
            pre_r_mobility = scipy.stats.pearsonr(y_obs_mobility, y_pred_mobility)[0]
            pre_mae_mobility = median_absolute_error(y_obs_mobility, y_pred_mobility)
            pre_r2_mobility = r2_score(y_obs_mobility, y_pred_mobility)
            
            pretrained_metrics = {'R2 (CCS)': pre_r2_ccs, 'MAE (CCS)': pre_mae_ccs, 'R2 (mobility)': pre_r2_mobility, 'MAE (mobility)': pre_mae_mobility}
            logging.info(f"Testing pretrained CCS model (CCS space): R2 (CCS)={pre_r2_ccs:.4f}, MAE (CCS)={pre_mae_ccs:.4f}, R2 (mobility)={pre_r2_mobility:.4f}, MAE (mobility)={pre_mae_mobility:.4f}")
            
            # Save pretrained results and generate plot
            test_psm_df.to_csv(os.path.join(out_dir, "mobility_test_pretrained.tsv"), sep="\t", index=False)
            plot_mobility(out_dir, test_file="mobility_test_pretrained.tsv", 
                          output_plot="mobility_performance_pretrained.png", title="Mobility Prediction Performance (Pretrained)")
        except Exception as e:
            logging.warning(f"Could not perform pre-training CCS evaluation: {e}")
    
    ccs_model._prepare_training(tr_df, lr_to_train_rt_ccs)
    lr_scheduler = ccs_model._get_lr_schedule_with_warmup(warmup_epoch_to_train_rt_ccs, epoch_to_train_rt_ccs)
    
    best_test_loss = [float('inf')]
    epochs_without_improvement = [0]
    stopped_early = [False]

    def run_training_loop():
        for ep in range(epoch_to_train_rt_ccs):
            batch_cost = ccs_model._train_one_epoch(tr_df, ep, batch_size_to_train_rt_ccs, False)
            lr_scheduler.step()
            train_loss = np.mean(batch_cost)
            
            test_loss_str = ""
            test_loss = 0
            test_losses = []
            if test_psm_df is not None and len(test_psm_df) > 0:
                ccs_model.model.eval()
                with torch.no_grad():
                    for nAA, df_group in test_psm_df.groupby('nAA', sort=False):
                        for i in range(0, len(df_group), batch_size_to_train_rt_ccs):
                            batch_df = df_group.iloc[i:i + batch_size_to_train_rt_ccs, :]
                            targets = ccs_model._get_targets_from_batch_df(batch_df)
                            features = ccs_model._get_features_from_batch_df(batch_df)
                            predicts = ccs_model.model(*features)
                            loss = torch.nn.L1Loss()(predicts, targets)
                            test_losses.append(loss.item())
                ccs_model.model.train()
                if test_losses:
                    test_loss = np.mean(test_losses)
                    test_loss_str = f", test_loss={test_loss}"
            
            ccs_model.history['epoch'].append(ep + 1)
            ccs_model.history['lr'].append(lr_scheduler.get_last_lr()[0])
            ccs_model.history['train_loss'].append(train_loss)
            ccs_model.history['test_loss'].append(test_loss if test_losses else 0)

            if test_losses:
                if test_loss < best_test_loss[0]:
                    best_test_loss[0] = test_loss
                    epochs_without_improvement[0] = 0
                    if use_best_model:
                        ccs_model.save(out_dir + "/ccs_model.pt")
                else:
                    epochs_without_improvement[0] += 1
            
            logging.info(f"[Training] Epoch={ep + 1}, lr={lr_scheduler.get_last_lr()[0]:.1e}, train_loss={train_loss}{test_loss_str}")
            
            if early_stop and epochs_without_improvement[0] >= patience:
                stopped_early[0] = True
                break
        torch.cuda.empty_cache()

    run_training_loop()
    if use_best_model and os.path.exists(out_dir + "/ccs_model.pt"):
        ccs_model.load(out_dir + "/ccs_model.pt")
    
    ccs_model.save(out_dir + "/ccs_model.pt")

    # Post-training evaluation
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import median_absolute_error,r2_score
            test_psm_df['ccs_pred'] = ccs_model.predict(test_psm_df.copy())['ccs_pred']
            # Also need mobility_pred for plotting
            test_psm_df['mobility_pred'] = ccs_model.ccs_to_mobility_pred(test_psm_df)
            
            y_obs_ccs = test_psm_df['ccs'].values
            y_pred_ccs = test_psm_df['ccs_pred'].values
            post_r_ccs = scipy.stats.pearsonr(y_obs_ccs, y_pred_ccs)[0]
            post_mae_ccs = median_absolute_error(y_obs_ccs, y_pred_ccs)
            post_r2_ccs = r2_score(y_obs_ccs, y_pred_ccs)
            
            y_obs_mobility = test_psm_df['mobility'].values
            y_pred_mobility = test_psm_df['mobility_pred'].values
            post_r_mobility = scipy.stats.pearsonr(y_obs_mobility, y_pred_mobility)[0]
            post_mae_mobility = median_absolute_error(y_obs_mobility, y_pred_mobility)
            post_r2_mobility = r2_score(y_obs_mobility, y_pred_mobility)
            
            eval_label = "Final Evaluation (Best Model)" if use_best_model else "Final Evaluation (Last Model)"
            logging.info(f"{eval_label}: R2 (CCS)={post_r2_ccs:.4f}, MAE (CCS)={post_mae_ccs:.4f}, R2 (mobility)={post_r2_mobility:.4f}, MAE (mobility)={post_mae_mobility:.4f}")
            
            if pretrained_metrics is not None:
                logging.info("\n=== Improvement Summary (CCS) ===")
                logging.info(f"{'Metric':<20} {'Pretrained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
                logging.info("-" * 60)
                for metric in ['R2 (CCS)', 'MAE (CCS)', 'R2 (mobility)', 'MAE (mobility)']:
                    pre = pretrained_metrics[metric]
                    if metric == 'R2 (CCS)':
                        post = post_r2_ccs
                    elif metric == 'MAE (CCS)':
                        post = post_mae_ccs
                    elif metric == 'R2 (mobility)':
                        post = post_r2_mobility
                    elif metric == 'MAE (mobility)':
                        post = post_mae_mobility
                    delta = post - pre
                    if metric.lower().startswith("r2"):
                        sign = "+" if delta >= 0 else "-"
                    elif metric.lower().startswith("mae"):
                        sign = "-" if delta >= 0 else "+"
                    else:
                        print(f"Unknown metric: {metric}")
                        continue
                    logging.info(f"{metric:<20} {pre:>12.4f} {post:>12.4f} {sign}{delta:>11.4f}")
            
            test_psm_df.to_csv(os.path.join(out_dir, "mobility_test.tsv"), sep="\t", index=False)
        except Exception as e:
            logging.warning(f"Could not perform post-training CCS evaluation: {e}")

    ccs_model.save_training_history(os.path.join(out_dir, "ccs_history.tsv"))
    ccs_model.plot_training_history(os.path.join(out_dir, "ccs_history.png"), mark_best=use_best_model)
    return model_mgr

def plot_rt(in_dir, out_dir, test_file="rt_test.tsv", 
            output_plot="rt_performance_test.png", 
            title="RT Prediction Performance",
            denormalize_rt=True):
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

    # Backwards compatibility for callers using positional args
    if test_file == "rt_test.tsv" and not os.path.exists(os.path.join(out_dir, test_file)):
         # If default not found, maybe it was passed in differently? 
         # But usually we'd expect it in out_dir.
         pass

    full_test_path = os.path.join(out_dir, test_file)
    if not os.path.exists(full_test_path):
        print(f"Test file {full_test_path} not found. Skipping plotting.")
        return

    # meta.json should be in in_dir
    meta_file = os.path.join(in_dir, "meta.json") if in_dir else None
    if meta_file and os.path.exists(meta_file):
        print(f"Found meta.json: {meta_file}")
    elif meta_file:
        print(f"meta.json not found in {in_dir}")
    
    rt_min = 0.0
    rt_max = 1.0
    if meta_file and os.path.exists(meta_file):
        try:
            # Use utf-8-sig to handle BOM which might cause JSON parsing 
            with open(meta_file, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                # Robust parsing: handle unescaped backslashes commonly found in Windows paths
                import re
                content = re.sub(r'(?<!\\)\\(?![\\"])', lambda m: r'\\', content)
                try:
                    meta_data = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback for very broken paths
                    with open(meta_file, 'r', encoding='utf-8-sig') as f2:
                         content = f2.read()
                    content = content.replace('\\', '\\\\').replace('\\\\"', '\\"')
                    meta_data = json.loads(content)
                
                if meta_data:
                    all_rt_min = []
                    all_rt_max = []
                    for key in meta_data:
                        if isinstance(meta_data[key], dict):
                            if 'rt_min' in meta_data[key]:
                                all_rt_min.append(meta_data[key]['rt_min'])
                            if 'rt_max' in meta_data[key]:
                                all_rt_max.append(meta_data[key]['rt_max'])
                    if all_rt_min: rt_min = min(all_rt_min)
                    if all_rt_max: rt_max = max(all_rt_max)
                    print(f"Using global RT normalization factors: rt_min={rt_min}, rt_max={rt_max}")
        except Exception as e:
            print(f"Error reading meta.json: {e}")

    try:
        df = pd.read_csv(full_test_path, sep="\t")
        if 'rt_norm' not in df.columns or 'rt_pred' not in df.columns:
            print(f"Columns 'rt_norm' or 'rt_pred' not found in {full_test_path}")
            return
            
        # Denormalize to minutes
        y_obs = df['rt_norm'] * (rt_max - rt_min) + rt_min
        if denormalize_rt:
            y_pred = df['rt_pred'] * (rt_max - rt_min) + rt_min
        else:
            y_pred = df['rt_pred']

        cor = scipy.stats.pearsonr(y_obs, y_pred)[0]
        mae = median_absolute_error(y_obs, y_pred)
        r2 = r2_score(y_obs, y_pred)
        range_95 = np.percentile(np.abs(y_obs - y_pred), 95)

        plt.rcParams['figure.figsize'] = [5, 5]
        plt.scatter(y_obs, y_pred, s=4, c="blue", alpha=0.5)
        
        max_rt = max(rt_max, y_obs.max(), y_pred.max())*1.02

        if denormalize_rt:
            plt.plot([0, max_rt], [0, max_rt], color='red', linestyle='--', linewidth=1)
        
        if denormalize_rt:
            stats_text = f"PCC = {cor:.4f}\nMAE = {mae:.2f} min\n$R^2$ = {r2:.4f}\n95% Range = {range_95:.2f} min\nN = {len(df)}"
        else:
            stats_text = f"PCC = {cor:.4f}\nN = {len(df)}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Observed RT (Minute)')
        if denormalize_rt:
            plt.ylabel('Predicted RT (Minute)')
        else:
            plt.ylabel('Predicted RT')
        plt.xlim(0, max_rt)
        if denormalize_rt:
            plt.ylim(0, max_rt)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.title(title)
        
        output_plot_path = os.path.join(out_dir, output_plot)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        print(f"RT performance plot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error generating RT plot: {e}")


def plot_mobility(out_dir, test_file="mobility_test.tsv", output_plot="mobility_performance_test.png", title="Mobility Prediction Performance"):
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

    full_test_path = os.path.join(out_dir, test_file)
    if not os.path.exists(full_test_path):
        print(f"Test file {full_test_path} not found. Skipping plotting.")
        return

    try:
        df = pd.read_csv(full_test_path, sep="\t")
        if 'mobility' not in df.columns or 'mobility_pred' not in df.columns:
            print(f"Columns 'mobility' or 'mobility_pred' not found in {full_test_path}")
            return
            
        y_obs = df['mobility']
        y_pred = df['mobility_pred']

        cor = scipy.stats.pearsonr(y_obs, y_pred)[0]
        mae = median_absolute_error(y_obs, y_pred)
        r2 = r2_score(y_obs, y_pred)
        range_95 = np.percentile(np.abs(y_obs - y_pred), 95)

        plt.rcParams['figure.figsize'] = [5, 5]
        plt.scatter(y_obs, y_pred, s=4, c="blue", alpha=0.5)
        
        max_mobility = max(y_obs.max(), y_pred.max())*1.02
        min_mobility = min(y_obs.min(), y_pred.min())*0.98
        plt.plot([min_mobility, max_mobility], [min_mobility, max_mobility], color='red', linestyle='--', linewidth=1)
        
        stats_text = f"PCC = {cor:.4f}\nMAE = {mae:.4f}\n$R^2$ = {r2:.4f}\n95% Range = {range_95:.4f}\nN = {len(df)}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel('Observed Mobility')
        plt.ylabel('Predicted Mobility')
        plt.xlim(min_mobility, max_mobility)
        plt.ylim(min_mobility, max_mobility)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.title(title)
        
        output_plot_path = os.path.join(out_dir, output_plot)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        print(f"Mobility performance plot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error generating mobility plot: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--out_prefix", type=str, default="")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--instrument", type=str, default="Astral")
    parser.add_argument("--tf_type", type=str, default="all")
    parser.add_argument("--nce", type=float, default=27.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mode", type=str, default="general")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument('--log_transform', action='store_true', help='log transform intensity data')
    parser.add_argument('--no_masking', action='store_true', help='disable masking for training')
    parser.add_argument('--user_mod', default=None, help='User defined modification')
    parser.add_argument('--use_grid_nce_search', action='store_true', help='Use grid search for NCE and instrument (default: False)')
    parser.add_argument('--use_best_model', action='store_true', help='Save model with lowest test loss (default: False, saves last epoch)')
    parser.add_argument('--auto_tune', action='store_true', help='Dynamic scaling of hyperparameters for 1k-100k range (default: False)')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping when validation loss stops improving (default: False)')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping (default: 10)')
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads to use")
    parser.add_argument("--torch_compile", action="store_true", help="Compile model with torch.compile for speed (CPU/GPU)")
    parser.add_argument("--profile", action='store_true', help="Profile the script execution")
    parser.add_argument("--ms2_model", type=str, default=None, help="Pretrained MS2 model path")
    parser.add_argument("--rt_model", type=str, default=None, help="Pretrained RT model path")
    parser.add_argument("--ccs_model", type=str, default=None, help="Pretrained CCS model path")
    args = parser.parse_args()

    # Early stopping requires best model checkpointing
    if args.early_stop and not args.use_best_model:
        args.use_best_model = True
        logging.info("Enabling --use_best_model since --early_stop is active")

    import psutil
    n_physical = psutil.cpu_count(logical=False) or 4
    
    if args.threads is None:
        args.threads = n_physical
        
    configure_env_for_device(args.device, args.threads)

    # Add models.py to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    print(f"Using local models.py from: {script_dir}")

    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg') # Set backend to non-interactive

    
    # Critical CPU tuning (matching ai.py)
    device = args.device.strip().lower()
    if device == "cpu":
        torch.set_num_threads(args.threads)
        # Re-enabling interop threads if physical core count is high
        if args.threads > 8:
            try:
                torch.set_num_interop_threads(2)
            except RuntimeError:
                pass

    if args.verbose >= 2:
        print("torch intra", torch.get_num_threads())
        print("torch interop", torch.get_num_interop_threads())
    
    # Set seed - MUST be after torch import, BEFORE check_models_from_docker (matching ai.py)
    set_seed(int(args.seed))
    
    if args.verbose >= 2:
        print_thread_state(tag="Thread/BLAS summary (startup)")
    
    from threadpoolctl import threadpool_limits, threadpool_info
    
    if args.user_mod is not None:
        from models import add_user_defined_modifications
        user_mod = args.user_mod
        user_mod = re.sub(r'^\"',"",user_mod)
        user_mod = re.sub(r'\"$',"",user_mod)
        umods = user_mod.split(";")
        mod_dict = {}
        # format: 597.1216,K,469.0266,C(19)H(17)Cl(2)N(3)O(5)S(1)
        # format: modification name,amino acid,monoisotopic mass,composition
        for umod in umods:
            um = umod.split(",")
            mod_name = um[0]+"@"+um[1]
            mod_dict[mod_name] = {}
            mod_dict[mod_name]["composition"] = um[3]
            mod_dict[mod_name]["modloss_composition"] = ""
        print("User defined modifications (Applied):")
        print(mod_dict)
        add_user_defined_modifications(mod_dict)



    tf_type = args.tf_type
    
    def run_main():
        if tf_type == "all" or tf_type == "test":
            if tf_type == "test":
                print("Test mode ...")
            train_rt(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                device=args.device, threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                verbose=args.verbose, torch_compile=args.torch_compile,
                                pretrained_model=args.rt_model)
            plot_rt(in_dir, out_dir)
            # CCS training if data exists and has sufficient samples
            ccs_data_path = os.path.join(in_dir, "ccs_train_data.tsv")
            if os.path.exists(ccs_data_path):
                import pandas as pd
                ccs_data = pd.read_csv(ccs_data_path, sep="\t")
                if ccs_data.shape[0] >= 100:
                    train_ccs(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                            device=args.device, threads=args.threads, use_best_model=args.use_best_model,
                                            auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                            verbose=args.verbose, torch_compile=args.torch_compile,
                                            pretrained_model=args.ccs_model)
                    plot_mobility(out_dir)
            train_ms2(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                log_transform=args.log_transform,
                                device=args.device, instrument=args.instrument, nce=args.nce,
                                use_valid=(not args.no_masking),
                                threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop,
                                patience=args.patience, verbose=args.verbose,
                                torch_compile=args.torch_compile,
                                pretrained_model=args.ms2_model)
        elif tf_type == "rt":
            train_rt(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                device=args.device, threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                verbose=args.verbose, torch_compile=args.torch_compile,
                                pretrained_model=args.rt_model)
            plot_rt(in_dir, out_dir)
        elif tf_type == "ms2":
            train_ms2(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                log_transform=args.log_transform,
                                device=args.device, instrument=args.instrument, nce=args.nce,
                                use_valid=(not args.no_masking),
                                threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop,
                                patience=args.patience, verbose=args.verbose,
                                torch_compile=args.torch_compile,
                                pretrained_model=args.ms2_model)
        elif tf_type == "ccs":
            train_ccs(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                device=args.device, threads=args.threads, use_best_model=args.use_best_model,
                                auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                verbose=args.verbose, torch_compile=args.torch_compile,
                                pretrained_model=args.ccs_model)
            plot_mobility(out_dir)

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
        ps.print_stats(30) # Print top 30 functions
        print(s.getvalue())
        # Also save to file
        prof_file = os.path.join(args.out_dir, "training_profile.prof")
        pr.dump_stats(prof_file)
        print(f"Detailed profile saved to {prof_file}")
    else:
        run_main()

    # Cleanup at exit (matching ai.py)
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


