"""
Carafe Models - Standalone Training and Prediction Module
Extracted from alphapeptdeep/peptdeep with modifications for early stopping support.

This module provides MS2, RT, and CCS prediction models independent of the peptdeep package.
Dependencies: torch, pandas, numpy, transformers, alphabase, tqdm
"""

import os
import logging
import math
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    import torch.cuda.amp as amp
except ImportError:
    amp = None

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from typing import List, Tuple, Union, IO
from zipfile import ZipFile
import urllib.request
import urllib.error
import ssl
import socket
import shutil

# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

global_settings = {
    'PEPTDEEP_HOME': os.path.expanduser('~/peptdeep'),
    'local_model_zip_name': 'pretrained_models.zip',
    'model_url': 'https://github.com/MannLabs/alphapeptdeep/releases/download/pre-trained-models/pretrained_models.zip',
    'model_mgr': {
        'default_instrument': 'Lumos',
        'default_nce': 30.0,
        'mask_modloss': True,
        'model_type': 'generic',
        'model_zip_path': '',
        'instrument_group': {
            'ThermoTOF': 'ThermoTOF', 'Astral': 'Lumos', 'Lumos': 'Lumos', 'QE': 'QE', 'timsTOF': 'timsTOF', 
            'SciexTOF': 'SciexTOF', 'Fusion': 'Lumos', 'Eclipse': 'Lumos', 'Velos': 'Lumos', 'Elite': 'Lumos', 
            'OrbitrapTribrid': 'Lumos', 'ThermoTribrid': 'Lumos', 'QE+': 'QE', 'QEHF': 'QE', 'QEHFX': 'QE', 
            'Exploris': 'QE', 'Exploris480': 'QE', 'THERMOTOF': 'ThermoTOF', 'ASTRAL': 'Lumos', 'LUMOS': 'Lumos', 
            'TIMSTOF': 'timsTOF', 'SCIEXTOF': 'SciexTOF', 'FUSION': 'Lumos', 'ECLIPSE': 'Lumos', 'VELOS': 'Lumos', 
            'ELITE': 'Lumos', 'ORBITRAPTRIBRID': 'Lumos', 'THERMOTRIBRID': 'Lumos', 'EXPLORIS': 'QE', 'EXPLORIS480': 'QE'
        }
    }
}

# Alias for compatibility
settings = global_settings

# HuggingFace BERT encoder for transformer models
from transformers.models.bert.modeling_bert import BertEncoder

# alphabase for modification and fragment handling
from alphabase.constants.modification import MOD_DF
from alphabase.peptide.fragment import (
    init_fragment_by_precursor_dataframe,
    update_sliced_fragment_dataframe,
    get_sliced_fragment_dataframe,
    get_charged_frag_types
)

from alphabase.peptide.precursor import is_precursor_sorted
from alphabase.peptide.mobility import ccs_to_mobility_for_df

# ============================================================================
# MODEL CONSTANTS
# ============================================================================

# Modification elements - MUST keep order for model compatibility
MOD_ELEMENTS = [
    'C', 'H', 'N', 'O', 'P', 'S',  # Basic elements (first 6)
    'B', 'F', 'I', 'K', 'U', 'V', 'W', 'X', 'Y',
    'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'Ba', 'Be', 'Bi', 'Bk', 'Br',
    'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Es',
    'Eu', 'Fe', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'He', 'Hf', 'Hg', 'Ho', 'In', 'Ir',
    'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 'Ne',
    'Ni', 'No', 'Np', 'Os', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra',
    'Rb', 'Re', 'Rh', 'Rn', 'Ru', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta',
    'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Xe', 'Yb', 'Zn', 'Zr',
    '2H', '13C', '15N', '18O', '?'  # Isotopes and unknown
]

INSTRUMENTS = ['QE', 'Lumos', 'timsTOF', 'SciexTOF', 'ThermoTOF']
MAX_INSTRUMENT_NUM = 8
AA_EMBEDDING_SIZE = 27

FRAG_TYPES = ['b', 'y', 'b_modloss', 'y_modloss']
MAX_FRAG_CHARGE = 2

mod_feature_size = len(MOD_ELEMENTS)
mod_elem_to_idx = dict(zip(MOD_ELEMENTS, range(mod_feature_size)))
num_ion_types = len(FRAG_TYPES) * MAX_FRAG_CHARGE

instrument_dict = {inst.upper(): i for i, inst in enumerate(INSTRUMENTS)}
unknown_inst_index = MAX_INSTRUMENT_NUM - 1

# ============================================================================
# SETTINGS (local replacement for peptdeep.settings)
# ============================================================================



# Settings dict used by model classes (replaces peptdeep.settings.global_settings)
settings = {
    'model_mgr': {
        'instrument_group': {
            # Map instrument names to their base types (matches peptdeep default_settings.yaml)
            'THERMOTOF': 'ThermoTOF',
            'ASTRAL': 'Lumos',
            'LUMOS': 'Lumos',
            'QE': 'QE',
            'TIMSTOF': 'timsTOF',
            'SCIEXTOF': 'SciexTOF',
            'FUSION': 'Lumos',
            'ECLIPSE': 'Lumos',
            'VELOS': 'Lumos',
            'ELITE': 'Lumos',
            'ORBITRAPTRIBRID': 'Lumos',
            'THERMOTRIBRID': 'Lumos',
            'QE+': 'QE',
            'QEHF': 'QE',
            'QEHFX': 'QE',
            'EXPLORIS': 'QE',
            'EXPLORIS480': 'QE',
        },
        'transfer': {
            'grid_nce_first': 15.0,
            'grid_nce_last': 45.0,
            'grid_nce_step': 3.0,
            'grid_instrument': ['Lumos'],
        }
    }
}

# Toggle between fast vectorized feature extraction (V2 behavior) and 
# iterative methods (Original PeptDeep behavior) for numerical parity.
USE_VECTORIZED_FEATURE_EXTRACTION = False 


def add_user_defined_modifications(user_mods: dict = None):
    """
    Add user-defined modifications into the system,
    this is useful for isotope labeling.

    Parameters
    ----------
    user_mods : dict, optional
        Example:
        ```
        {
        "Dimethyl2@Any N-term": { 
        "composition": "H(2)2H(2)C(2)",
        "modloss_composition": ""
        }, ...
        }
        ```
        By default None.
    """
    if user_mods is None:
        return
    
    from alphabase.constants.modification import add_new_modifications
    add_new_modifications(user_mods)
    
    # Update local feature cache
    update_all_mod_features()
    
    for mod_name in user_mods.keys():
        print(f"Added modification: {mod_name}")

# ============================================================================
# UTILITY FUNCTIONS (extracted from peptdeep)
# ============================================================================

def download_models(url: str = global_settings['model_url'], overwrite=True):
    """
    Download pretrained models from a remote URL.
    Implementation mirrored from peptdeep.pretrained_models.
    """
    model_zip_name = global_settings['local_model_zip_name']
    pretrain_dir = os.path.join(global_settings['PEPTDEEP_HOME'], "pretrained_models")
    
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
        
    model_zip = os.path.join(pretrain_dir, model_zip_name)
    
    if not os.path.isfile(url):
        logging.info(f'Downloading {model_zip_name} to {model_zip} ...')
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context, timeout=10) as response:
                with open(model_zip, 'wb') as f:
                    f.write(response.read())
        except (socket.timeout, urllib.error.URLError, urllib.error.HTTPError) as e:
            logging.error(f'Downloading model failed: {e}')
            print('Downloading model failed! Please download the '
                  f'zip file by yourself from "{url}",'
                  f' and place it at "{model_zip}"')
            return None
    else:
        shutil.copy(url, model_zip)
    
    logging.info(f'Pretrained models downloaded to {model_zip}')
    return model_zip


def linear_regression(x, y):
    """Compute linear regression between x and y"""
    coeffs = np.polyfit(x, y, 1)
    w, b = coeffs.tolist()
    yhat = np.poly1d(coeffs)(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    R_square = ssreg / sstot if sstot > 0 else 0
    return dict(
        R_square=[R_square],
        R=[np.sqrt(R_square)],
        slope=[w],
        intercept=[b],
    )

def evaluate_linear_regression(df: pd.DataFrame, x='rt_pred', y='rt_norm', n_sample=10000000):
    """Evaluate RT prediction with linear regression metrics"""
    if len(df) > n_sample:
        df = df.sample(n_sample, replace=False)
    regs = linear_regression(df[x].values, df[y].values)
    regs["test_num"] = len(df)
    return pd.DataFrame(regs)

def uniform_sampling(psm_df: pd.DataFrame, target: str = 'rt_norm', n_train: int = 1000, random_state: int = 1337) -> pd.DataFrame:
    """Divide psm_df into 10 bins and sample training values from each bin"""
    x = np.arange(0, 11) / 10 * psm_df[target].max()
    sub_n = n_train // (len(x) - 1)
    df_list = []
    for i in range(len(x) - 1):
        _df = psm_df[(psm_df[target] >= x[i]) & (psm_df[target] < x[i + 1])]
        if len(_df) == 0:
            pass
        elif len(_df) // 2 < sub_n:
            df_list.append(_df.sample(len(_df) // 2, replace=False, random_state=random_state))
        else:
            df_list.append(_df.sample(sub_n, replace=False, random_state=random_state))
    if len(df_list) == 0:
        return pd.DataFrame()
    return pd.concat(df_list)

def count_mods(psm_df: pd.DataFrame) -> pd.DataFrame:
    """Count modifications in PSM dataframe"""
    mods = psm_df[psm_df.mods.str.len() > 0].mods.apply(lambda x: x.split(';'))
    mod_dict = {}
    mod_dict['mutation'] = {'spec_count': 0}
    for one_mods in mods.values:
        for mod in set(one_mods):
            items = mod.split('->')
            if len(items) == 2 and len(items[0]) == 3 and len(items[1]) == 5:
                mod_dict['mutation']['spec_count'] += 1
            elif mod not in mod_dict:
                mod_dict[mod] = {'spec_count': 1}
            else:
                mod_dict[mod]['spec_count'] += 1
    return pd.DataFrame.from_dict(mod_dict, orient='index').reset_index(drop=False).rename(
        columns={'index': 'mod'}).sort_values('spec_count', ascending=False).reset_index(drop=True)

def psm_sampling_with_important_mods(psm_df, n_sample, top_n_mods=10, n_sample_each_mod=0,
                                      uniform_sampling_column=None, random_state=1337):
    """Sample PSMs with important modifications for training"""
    if n_sample >= len(psm_df):
        return psm_df.copy()

    psm_df_list = []
    if uniform_sampling_column is None:
        def _sample(psm_df, n):
            if n < len(psm_df):
                return psm_df.sample(n, replace=False, random_state=random_state).copy()
            else:
                return psm_df.copy()
    else:
        def _sample(psm_df, n):
            if len(psm_df) == 0:
                return psm_df
            return uniform_sampling(psm_df, target=uniform_sampling_column, n_train=n, random_state=random_state)

    psm_df_list.append(_sample(psm_df, n_sample))
    if n_sample_each_mod > 0:
        mod_df = count_mods(psm_df)
        mod_df = mod_df[mod_df['mod'] != 'mutation']
        if len(mod_df) > top_n_mods:
            mod_df = mod_df.iloc[:top_n_mods, :]
        for mod in mod_df['mod'].values:
            psm_df_list.append(_sample(psm_df[psm_df.mods.str.contains(mod, regex=False, na=False)], n_sample_each_mod))
    if len(psm_df_list) > 0:
        return pd.concat(psm_df_list, ignore_index=True).copy()
    else:
        return pd.DataFrame()

def normalize_fragment_intensities(psm_df: pd.DataFrame, frag_intensity_df: pd.DataFrame):
    """Normalize intensities to 0-1 values inplace"""
    frag_intensity_df_np = frag_intensity_df.to_numpy()
    for i, (frag_start_idx, frag_stop_idx) in enumerate(psm_df[['frag_start_idx', 'frag_stop_idx']].values):
        intens = frag_intensity_df_np[frag_start_idx:frag_stop_idx]
        max_inten = np.max(intens)
        if max_inten > 0:
            intens /= max_inten
        frag_intensity_df.iloc[frag_start_idx:frag_stop_idx, :] = intens

def pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute pearson correlation between 2 batches of 1-D tensors"""
    return torch.cosine_similarity(x - x.mean(dim=1, keepdim=True), y - y.mean(dim=1, keepdim=True), dim=1)

def pearson_correlation_mask(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    """Compute pearson correlation with mask"""
    x_masked = x * mask
    y_masked = y * mask
    x_mean = x_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
    y_mean = y_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
    return torch.cosine_similarity((x - x_mean) * mask, (y - y_mean) * mask, dim=1)

def spectral_angle(cos):
    """Compute spectral angle from cosine similarity"""
    cos = cos.clone()
    cos[cos > 1] = 1
    return 1 - 2 * torch.arccos(cos) / np.pi

def _get_ranks(x: torch.Tensor, device) -> torch.Tensor:
    """Get ranks for spearman correlation"""
    sorted_idx = x.argsort(dim=1)
    flat_idx = (sorted_idx + torch.arange(x.size(0), device=device).unsqueeze(1) * x.size(1)).flatten()
    ranks = torch.zeros_like(flat_idx)
    ranks[flat_idx] = torch.arange(x.size(1), device=device).unsqueeze(0).repeat(x.size(0), 1).flatten()
    ranks = ranks.reshape(x.size())
    ranks[x == 0] = 0
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor, device):
    """Compute spearman correlation between 2 batches of 1-D tensors"""
    x_rank = _get_ranks(x, device).to(torch.float32)
    y_rank = _get_ranks(y, device).to(torch.float32)
    n = x.size(1)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=1)
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def spearman_correlation_mask(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, device):
    """Compute spearman correlation between 2 batches of 1-D tensors with mask"""
    x_rank = _get_ranks(x, device).to(torch.float32)
    y_rank = _get_ranks(y, device).to(torch.float32)
    
    # We use Pearson correlation logic on ranks while respecting the mask.
    # The mask ensures we only consider valid fragment indices.
    return pearson_correlation_mask(x_rank, y_rank, mask)

def add_cutoff_metric(metrics_describ, metrics_df, thres=0.9):
    """Add cutoff metric to description dataframe"""
    vals = []
    for col in metrics_describ.columns.values:
        vals.append(metrics_df.loc[metrics_df[col] > thres, col].count() / len(metrics_df))
    metrics_describ.loc[f'>{thres:.2f}'] = vals
    return metrics_describ

def calc_ms2_similarity(psm_df: pd.DataFrame, predict_intensity_df: pd.DataFrame,
                        fragment_intensity_df: pd.DataFrame, charged_frag_types: List = None,
                        metrics=['PCC', 'COS', 'SA', 'SPC'], GPU=True, batch_size=10240,
                        verbose=False, spc_top_k=0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate MS2 similarity metrics (PCC, COS, SA, SPC)"""
    if GPU:
        device, _ = get_available_device()
    else:
        device = torch.device('cpu')

    if charged_frag_types is None or len(charged_frag_types) == 0:
        charged_frag_types = fragment_intensity_df.columns.values

    _grouped = psm_df.groupby('nAA')
    batch_tqdm = tqdm(_grouped) if verbose else _grouped

    for met in metrics:
        psm_df[met] = 0

    for nAA, df_group in batch_tqdm:
        for i in range(0, len(df_group), batch_size):
            batch_end = i + batch_size
            batch_df = df_group.iloc[i:batch_end, :]

            pred_intens = torch.tensor(
                get_sliced_fragment_dataframe(predict_intensity_df, batch_df[['frag_start_idx', 'frag_stop_idx']].values, charged_frag_types).values,
                dtype=torch.float32, device=device
            ).reshape(-1, int((nAA - 1) * len(charged_frag_types)))

            frag_intens = torch.tensor(
                get_sliced_fragment_dataframe(fragment_intensity_df, batch_df[['frag_start_idx', 'frag_stop_idx']].values, charged_frag_types).values,
                dtype=torch.float32, device=device
            ).reshape(-1, int((nAA - 1) * len(charged_frag_types)))

            if 'PCC' in metrics:
                psm_df.loc['PCC'] = np.NaN
                psm_df.loc[batch_df.index, 'PCC'] = pearson_correlation(pred_intens, frag_intens).cpu().detach().numpy().astype(np.float32)
            if 'COS' in metrics or 'SA' in metrics:
                cos = torch.cosine_similarity(pred_intens, frag_intens, dim=1)
                psm_df.loc['COS'] = np.NaN
                psm_df.loc[batch_df.index, 'COS'] = cos.cpu().detach().numpy().astype(np.float32)
                if 'SA' in metrics:
                    psm_df.loc['SA'] = np.NaN
                    psm_df.loc[batch_df.index, 'SA'] = spectral_angle(cos).cpu().detach().numpy().astype(np.float32)
            if 'SPC' in metrics:
                psm_df.loc['SPC'] = np.NaN
                # Apply top-k filtering for spearman correlation if specified
                # Apply top-k filtering for spearman correlation if specified
                pred_for_spc = pred_intens
                frag_for_spc = frag_intens
                if spc_top_k > 1 and spc_top_k < frag_intens.size(1):
                    sorted_idx = frag_intens.argsort(dim=1, descending=True)
                    flat_idx = (
                        sorted_idx[:, :spc_top_k] + torch.arange(
                            frag_intens.size(0), dtype=torch.int,
                            device=device
                        ).unsqueeze(1) * frag_intens.size(1)
                    ).flatten()
                    pred_for_spc = pred_intens.flatten()[flat_idx].reshape(
                        sorted_idx.size(0), -1
                    )
                    frag_for_spc = frag_intens.flatten()[flat_idx].reshape(
                        sorted_idx.size(0), -1
                    )
                psm_df.loc[batch_df.index, 'SPC'] = spearman_correlation(
                    pred_for_spc, frag_for_spc, device
                ).cpu().detach().numpy().astype(np.float32)

    metrics_describ = psm_df[metrics].describe()
    add_cutoff_metric(metrics_describ, psm_df, thres=0.9)
    add_cutoff_metric(metrics_describ, psm_df, thres=0.75)
    torch.cuda.empty_cache()
    return psm_df, metrics_describ

def calc_ms2_similarity_mask(psm_df: pd.DataFrame, predict_intensity_df: pd.DataFrame,
                              fragment_intensity_df: pd.DataFrame, fragment_intensity_valid_df: pd.DataFrame,
                              charged_frag_types: List = None, metrics=['PCC', 'COS', 'SA', 'SPC'],
                              GPU=True, batch_size=10240, verbose=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate MS2 similarity metrics with masking for valid fragments"""
    if GPU:
        device, _ = get_available_device()
    else:
        device = torch.device('cpu')

    if charged_frag_types is None or len(charged_frag_types) == 0:
        charged_frag_types = fragment_intensity_df.columns.values

    _grouped = psm_df.groupby('nAA')
    batch_tqdm = tqdm(_grouped) if verbose else _grouped

    for met in metrics:
        psm_df[met] = 0

    for nAA, df_group in batch_tqdm:
        for i in range(0, len(df_group), batch_size):
            batch_end = i + batch_size
            batch_df = df_group.iloc[i:batch_end, :]

            pred_intens = torch.tensor(
                get_sliced_fragment_dataframe(predict_intensity_df, batch_df[['frag_start_idx', 'frag_stop_idx']].values, charged_frag_types).values,
                dtype=torch.float32, device=device
            ).reshape(-1, int((nAA - 1) * len(charged_frag_types)))

            frag_intens = torch.tensor(
                get_sliced_fragment_dataframe(fragment_intensity_df, batch_df[['frag_start_idx', 'frag_stop_idx']].values, charged_frag_types).values,
                dtype=torch.float32, device=device
            ).reshape(-1, int((nAA - 1) * len(charged_frag_types)))

            if fragment_intensity_valid_df is not None:
                frag_intens_valid = torch.tensor(
                    get_sliced_fragment_dataframe(fragment_intensity_valid_df, batch_df[['frag_start_idx', 'frag_stop_idx']].values, charged_frag_types).values,
                    dtype=torch.float32, device=device
                ).reshape(-1, int((nAA - 1) * len(charged_frag_types)))

                mask = torch.where(frag_intens_valid <= 0, 1, 0)
            else:
                mask = torch.ones(frag_intens.shape, dtype=torch.float32, device=device)

            if 'PCC' in metrics:
                psm_df.loc['PCC'] = np.NaN
                psm_df.loc[batch_df.index, 'PCC'] = pearson_correlation_mask(pred_intens, frag_intens, mask).cpu().detach().numpy().astype(np.float32)
            if 'COS' in metrics or 'SA' in metrics:
                cos = torch.cosine_similarity(pred_intens * mask, frag_intens * mask, dim=1)
                psm_df.loc['COS'] = np.NaN
                psm_df.loc[batch_df.index, 'COS'] = cos.cpu().detach().numpy().astype(np.float32)
                if 'SA' in metrics:
                    psm_df.loc['SA'] = np.NaN
                    psm_df.loc[batch_df.index, 'SA'] = spectral_angle(cos).cpu().detach().numpy().astype(np.float32)
            if 'SPC' in metrics:
                psm_df.loc['SPC'] = np.NaN
                psm_df.loc[batch_df.index, 'SPC'] = spearman_correlation_mask(pred_intens, frag_intens, mask, device).cpu().detach().numpy().astype(np.float32)

    metrics_describ = psm_df[metrics].describe()
    add_cutoff_metric(metrics_describ, psm_df, thres=0.9)
    add_cutoff_metric(metrics_describ, psm_df, thres=0.75)
    torch.cuda.empty_cache()
    return psm_df, metrics_describ

# ============================================================================
# FEATURIZATION FUNCTIONS
# ============================================================================

def _parse_mod_formula(formula):
    """Parse a modification formula to a feature vector"""
    feature = np.zeros(mod_feature_size)
    elems = formula.strip(')').split(')')
    for elem in elems:
        if '(' not in elem:
            continue
        chem, num = elem.split('(')
        num = int(num)
        if chem in mod_elem_to_idx:
            feature[mod_elem_to_idx[chem]] = num
        else:
            feature[-1] += num
    return feature

MOD_TO_FEATURE = {}
MOD_TO_ID = {}
MOD_FEATURE_MATRIX = None

def update_all_mod_features():
    """Update modification feature dictionary from alphabase MOD_DF"""
    global MOD_FEATURE_MATRIX
    MOD_TO_FEATURE.clear()
    MOD_TO_ID.clear()
    
    temp_feats = []
    for modname, formula in MOD_DF[['mod_name', 'composition']].values:
        if pd.notna(formula) and formula:
            feat = _parse_mod_formula(formula)
            MOD_TO_FEATURE[modname] = feat
            MOD_TO_ID[modname] = len(temp_feats)
            temp_feats.append(feat)
    
    # Add an empty feature for unknown modifications at the end
    MOD_FEATURE_MATRIX = np.vstack(temp_feats + [np.zeros(mod_feature_size)])
    # Unknown mods map to the last index
    MOD_TO_ID[''] = len(temp_feats) 

update_all_mod_features()

def get_batch_aa_indices(batch_df: pd.DataFrame) -> np.ndarray:
    """Convert peptide sequences into AA ID array (1-26 for A-Z, 0 for padding)"""
    # Use pre-calculated indices from the DataFrame
    if '_aa_indices' in batch_df.columns:
        aas = np.stack(batch_df._aa_indices.values)
    else:
        # Fallback if not pre-parsed (though it should be)
        seq_array = batch_df['sequence'].values.astype('U')
        aas = np.array(seq_array).view(np.int32).reshape(len(seq_array), -1) - ord('A') + 1
        
    return np.pad(aas, [(0, 0), (1, 1)])

def get_batch_mod_feature(batch_df: pd.DataFrame) -> np.ndarray:
    """Get modification features for a batch of peptides using vectorized indices"""
    nAA = int(batch_df.nAA.values[0])
    mod_x_batch = np.zeros((len(batch_df), nAA + 2, mod_feature_size))
    
    # Check if we should use the vectorized fast path
    if USE_VECTORIZED_FEATURE_EXTRACTION and '_mod_site_list' in batch_df.columns:
        # FAST Vectorized collection
        # We use list(values) to get the underlying objects efficiently
        mod_id_lists = batch_df._mod_id_list.values
        mod_site_lists = batch_df._mod_site_list.values
        
        # Filter out empty lists to speed up concatenate
        active_mask = [len(x) > 0 for x in mod_id_lists]
        if not any(active_mask):
            return mod_x_batch
            
        # Get IDs and Sites for the whole batch at once
        mod_ids = np.concatenate(mod_id_lists[active_mask])
        site_indices = np.concatenate(mod_site_lists[active_mask])
        
        # Calculate row indices for np.add.at
        # We repeat the row number (0, 1, 2...) total_mods[row] times
        lengths = [len(x) for x in mod_id_lists[active_mask]]
        active_rows = np.where(active_mask)[0]
        row_indices = np.repeat(active_rows, lengths)
        
        # Vectorized assignment: Add modification vectors to the batch tensor
        np.add.at(mod_x_batch, (row_indices, site_indices), MOD_FEATURE_MATRIX[mod_ids])
        return mod_x_batch

    # Fallback/Default to Iterative (Slow) Path
    # This matches PeptDeep's original summation order EXACTLY for numerical parity
    if '_mod_site_list' in batch_df.columns:
        # Use pre-parsed lists but iterate over them to match summation order
        mod_features_list = [
            [MOD_FEATURE_MATRIX[i] for i in ids] 
            for ids in batch_df._mod_id_list.values
        ]
        mod_sites_list = batch_df._mod_site_list.values
    else:
        # Original string parsing logic (if pre-parsing skipped)
        mod_features_list = batch_df.mods.str.split(';').apply(
            lambda mod_names: [MOD_TO_FEATURE.get(mod, np.zeros(mod_feature_size)) 
                                for mod in mod_names if len(mod) > 0]
        )
        mod_sites_list = batch_df.mod_sites.str.split(';').apply(
            lambda mod_sites: [int(site) for site in mod_sites if len(site) > 0]
        )

    for i, (mod_feats, mod_sites) in enumerate(zip(mod_features_list, mod_sites_list)):
        if len(mod_sites) > 0:
            for site, feat in zip(mod_sites, mod_feats):
                mod_x_batch[i, site, :] += feat
    return mod_x_batch

        
    return mod_x_batch

def parse_instrument_indices(instrument_list):
    """Map instrument names to indices"""
    return [instrument_dict.get(inst.upper(), unknown_inst_index) for inst in instrument_list]

# ============================================================================
# LR SCHEDULING
# ============================================================================

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """Create cosine LR schedule with warmup"""
    lr_lambda = functools.partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device(device_type: str = 'gpu', device_ids: list = []):
    """Get torch device based on type"""
    if device_type.lower() in ('gpu', 'cuda'):
        if torch.cuda.is_available():
            if device_ids:
                return torch.device(f'cuda:{device_ids[0]}'), 'cuda'
            return torch.device('cuda'), 'cuda'
    elif device_type.lower() == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), 'mps'
    return torch.device('cpu'), 'cpu'

def get_available_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), 'mps'
    return torch.device('cpu'), 'cpu'

# ============================================================================
# NEURAL NETWORK BUILDING BLOCKS
# ============================================================================

def aa_embedding(hidden_size):
    return nn.Embedding(AA_EMBEDDING_SIZE, hidden_size, padding_idx=0)

def ascii_embedding(hidden_size):
    return nn.Embedding(128, hidden_size, padding_idx=0)

def aa_one_hot(aa_indices, *cat_others):
    aa_x = torch.nn.functional.one_hot(aa_indices, AA_EMBEDDING_SIZE)
    return torch.cat((aa_x, *cat_others), 2)

def instrument_embedding(hidden_size):
    return nn.Embedding(MAX_INSTRUMENT_NUM, hidden_size)


def invert_attention_mask(encoder_attention_mask: torch.Tensor, dtype=torch.float32):
    """Invert attention mask for BERT encoder"""
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
    return encoder_extended_attention_mask


class SeqLSTM(nn.Module):
    """Bidirectional LSTM for sequence processing"""
    def __init__(self, in_features, out_features, rnn_layer=2, bidirectional=True):
        super().__init__()
        if bidirectional:
            if out_features % 2 != 0:
                raise ValueError("'out_features' must be divisible by 2")
            hidden = out_features // 2
        else:
            hidden = out_features
        self.rnn_h0 = nn.Parameter(torch.zeros(rnn_layer + rnn_layer * bidirectional, 1, hidden), requires_grad=False)
        self.rnn_c0 = nn.Parameter(torch.zeros(rnn_layer + rnn_layer * bidirectional, 1, hidden), requires_grad=False)
        self.rnn = nn.LSTM(
            input_size=in_features, hidden_size=hidden, num_layers=rnn_layer,
            batch_first=True, bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor):
        # Optimization: Use expand() instead of repeat() to avoid memory copies
        h0 = self.rnn_h0.expand(-1, x.size(0), -1).contiguous()
        c0 = self.rnn_c0.expand(-1, x.size(0), -1).contiguous()
        # Optimization: Ensure internal weights are correctly laid out for MKL/oneDNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x, (h0, c0))
        return x


class SeqAttentionSum(nn.Module):
    """Attention-based sequence aggregation"""
    def __init__(self, in_features):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_features, 1, bias=False), nn.Softmax(dim=1))

    def forward(self, x):
        attn = self.attn(x)
        return torch.sum(torch.mul(x, attn), dim=1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, out_features=128, max_len=200):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_features, 2) * (-np.log(max_len) / out_features))
        pe = torch.zeros(1, max_len, out_features)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _Pseudo_Bert_Config:
    """Pseudo config for HuggingFace BertEncoder"""
    def __init__(self, hidden_dim=256, intermediate_size=1024, num_attention_heads=8,
                 num_bert_layers=4, dropout=0.1, output_attentions=False):
        self.add_cross_attention = False
        self.chunk_size_feed_forward = 0
        self.is_decoder = False
        self.seq_len_dim = 1
        self.training = False
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = dropout
        self.attention_probs_dropout_prob = dropout
        self.hidden_size = hidden_dim
        self.initializer_range = 0.02
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = 1e-8
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_bert_layers
        self.output_attentions = output_attentions
        self._attn_implementation = "eager"


class Hidden_HFace_Transformer(nn.Module):
    """HuggingFace BERT-based transformer encoder"""
    def __init__(self, hidden_dim, hidden_expand=4, nheads=8, nlayers=4, dropout=0.1, output_attentions=False):
        super().__init__()
        self.config = _Pseudo_Bert_Config(
            hidden_dim=hidden_dim, intermediate_size=hidden_dim * hidden_expand,
            num_attention_heads=nheads, num_bert_layers=nlayers,
            dropout=dropout, output_attentions=False
        )
        self.output_attentions = output_attentions
        self.bert = BertEncoder(self.config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        if attention_mask is not None:
            attention_mask = invert_attention_mask(attention_mask, dtype=x.dtype)
        return self.bert(x, attention_mask=attention_mask, output_attentions=self.output_attentions, return_dict=False)


class Meta_Embedding(nn.Module):
    """Encode charge, NCE, and instrument into meta features"""
    def __init__(self, out_features):
        super().__init__()
        self.nn = nn.Linear(MAX_INSTRUMENT_NUM + 1, out_features - 1)

    def forward(self, charges, NCEs, instrument_indices):
        inst_x = torch.nn.functional.one_hot(instrument_indices, MAX_INSTRUMENT_NUM)
        meta_x = self.nn(torch.cat((inst_x.float(), NCEs), 1))
        return torch.cat((meta_x, charges), 1)


class Mod_Embedding_FixFirstK(nn.Module):
    """Modification embedding with fixed first k features"""
    def __init__(self, out_features):
        super().__init__()
        self.k = 6
        self.nn = nn.Linear(mod_feature_size - self.k, out_features - self.k, bias=False)

    def forward(self, mod_x):
        return torch.cat((mod_x[:, :, :self.k], self.nn(mod_x[:, :, self.k:])), 2)


class Input_26AA_Mod_PositionalEncoding(nn.Module):
    """Encode AA (26 letters) and modifications with positional encoding"""
    def __init__(self, out_features, max_len=200):
        super().__init__()
        mod_hidden = 8
        self.mod_nn = Mod_Embedding_FixFirstK(mod_hidden)
        self.aa_emb = aa_embedding(out_features - mod_hidden)
        self.pos_encoder = PositionalEncoding(out_features, max_len)

    def forward(self, aa_indices, mod_x):
        mod_x = self.mod_nn(mod_x)
        x = self.aa_emb(aa_indices)
        return self.pos_encoder(torch.cat((x, mod_x), 2))


class SeqCNN(nn.Module):
    """Multi-kernel CNN for sequence feature extraction"""
    def __init__(self, embedding_hidden):
        super().__init__()
        self.cnn_short = nn.Conv1d(embedding_hidden, embedding_hidden, kernel_size=3, padding=1)
        self.cnn_medium = nn.Conv1d(embedding_hidden, embedding_hidden, kernel_size=5, padding=2)
        self.cnn_long = nn.Conv1d(embedding_hidden, embedding_hidden, kernel_size=7, padding=3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.cnn_short(x)
        x2 = self.cnn_medium(x)
        x3 = self.cnn_long(x)
        return torch.cat((x, x1, x2, x3), dim=1).transpose(1, 2)


class Encoder_26AA_Mod_CNN_LSTM_AttnSum(nn.Module):
    """Encode AAs and modifications with CNN+LSTM and attention aggregation"""
    def __init__(self, out_features, n_lstm_layers=2):
        super().__init__()
        mod_hidden = 8
        self.mod_nn = Mod_Embedding_FixFirstK(mod_hidden)
        input_dim = AA_EMBEDDING_SIZE + mod_hidden
        self.input_cnn = SeqCNN(input_dim)
        self.hidden_nn = SeqLSTM(input_dim * 4, out_features, rnn_layer=n_lstm_layers)
        self.attn_sum = SeqAttentionSum(out_features)

    def forward(self, aa_indices, mod_x):
        mod_x = self.mod_nn(mod_x)
        x = aa_one_hot(aa_indices, mod_x)
        x = self.input_cnn(x)
        x = self.hidden_nn(x)
        return self.attn_sum(x)


class Encoder_26AA_Mod_Charge_CNN_LSTM_AttnSum(nn.Module):
    """Encode AAs, modifications, and charge with CNN+LSTM"""
    def __init__(self, out_features):
        super().__init__()
        mod_hidden = 8
        self.mod_nn = Mod_Embedding_FixFirstK(mod_hidden)
        input_dim = AA_EMBEDDING_SIZE + mod_hidden + 1
        self.input_cnn = SeqCNN(input_dim)
        self.hidden_nn = SeqLSTM(input_dim * 4, out_features, rnn_layer=2)
        self.attn_sum = SeqAttentionSum(out_features)

    def forward(self, aa_indices, mod_x, charges):
        mod_x = self.mod_nn(mod_x)
        x = aa_one_hot(aa_indices, mod_x, charges.unsqueeze(1).repeat(1, mod_x.size(1), 1))
        x = self.input_cnn(x)
        x = self.hidden_nn(x)
        return self.attn_sum(x)


class Decoder_Linear(nn.Module):
    """Linear decoder with hidden layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(in_features, 64), nn.PReLU(), nn.Linear(64, out_features))

    def forward(self, x):
        return self.nn(x)

# Aliases for compatibility
AATransformerEncoding = Input_26AA_Mod_PositionalEncoding

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ModelMS2Bert(nn.Module):
    """BERT-based MS2 prediction model"""
    def __init__(self, num_frag_types, num_modloss_types=0, mask_modloss=True,
                 dropout=0.1, nlayers=4, hidden=256, output_attentions=False, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self._num_modloss_types = num_modloss_types
        self._num_non_modloss = num_frag_types - num_modloss_types
        self._mask_modloss = mask_modloss if num_modloss_types > 0 else True

        meta_dim = 8
        self.input_nn = Input_26AA_Mod_PositionalEncoding(hidden - meta_dim)
        self.meta_nn = Meta_Embedding(meta_dim)
        self._output_attentions = output_attentions
        self.hidden_nn = Hidden_HFace_Transformer(hidden, nlayers=nlayers, dropout=dropout,
                                                   output_attentions=output_attentions)
        self.output_nn = Decoder_Linear(hidden, self._num_non_modloss)

        if num_modloss_types > 0:
            self.modloss_nn = nn.ModuleList([
                Hidden_HFace_Transformer(hidden, nlayers=1, dropout=dropout),
                Decoder_Linear(hidden, num_modloss_types)
            ])
        else:
            self.modloss_nn = None

    def forward(self, aa_indices, mod_x, charges, NCEs, instrument_indices):
        in_x = self.dropout(self.input_nn(aa_indices, mod_x))
        meta_x = self.meta_nn(charges, NCEs, instrument_indices).unsqueeze(1).repeat(1, in_x.size(1), 1)
        in_x = torch.cat((in_x, meta_x), 2)

        hidden_x = self.hidden_nn(in_x)
        self.attentions = hidden_x[1] if self._output_attentions else None
        hidden_x = self.dropout(hidden_x[0] + in_x * 0.2)
        out_x = self.output_nn(hidden_x)

        if self._num_modloss_types > 0:
            if self._mask_modloss:
                out_x = torch.cat((out_x, torch.zeros(*out_x.size()[:2], self._num_modloss_types, device=in_x.device)), 2)
            else:
                modloss_x = self.modloss_nn[0](in_x)[0] + hidden_x
                modloss_x = self.modloss_nn[-1](modloss_x)
                out_x = torch.cat((out_x, modloss_x), 2)
        return out_x[:, 3:, :]


class Model_RT_Bert(nn.Module):
    """BERT-based RT prediction model"""
    def __init__(self, dropout=0.1, nlayers=4, hidden=128, output_attentions=False, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_nn = Input_26AA_Mod_PositionalEncoding(hidden)
        self._output_attentions = output_attentions
        self.hidden_nn = Hidden_HFace_Transformer(hidden, nlayers=nlayers, dropout=dropout,
                                                   output_attentions=output_attentions)
        self.output_nn = nn.Sequential(SeqAttentionSum(hidden), nn.PReLU(), self.dropout, nn.Linear(hidden, 1))

    def forward(self, aa_indices, mod_x):
        x = self.dropout(self.input_nn(aa_indices, mod_x))
        hidden_x = self.hidden_nn(x)
        self.attentions = hidden_x[1] if self._output_attentions else None
        x = self.dropout(hidden_x[0] + x * 0.2)
        return self.output_nn(x).squeeze(1)


class Model_RT_LSTM_CNN(nn.Module):
    """CNN+LSTM RT prediction model"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        hidden = 256
        self.rt_encoder = Encoder_26AA_Mod_CNN_LSTM_AttnSum(hidden)
        self.rt_decoder = Decoder_Linear(hidden, 1)

    def forward(self, aa_indices, mod_x):
        x = self.rt_encoder(aa_indices, mod_x)
        x = self.dropout(x)
        return self.rt_decoder(x).squeeze(1)


class Model_CCS_Bert(nn.Module):
    """BERT-based CCS prediction model"""
    def __init__(self, dropout=0.1, nlayers=4, hidden=128, output_attentions=False, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_nn = Input_26AA_Mod_PositionalEncoding(hidden - 2)
        self._output_attentions = output_attentions
        self.hidden_nn = Hidden_HFace_Transformer(hidden, nlayers=nlayers, dropout=dropout,
                                                   output_attentions=output_attentions)
        self.output_nn = nn.Sequential(SeqAttentionSum(hidden), nn.PReLU(), self.dropout, nn.Linear(hidden, 1))

    def forward(self, aa_indices, mod_x, charges):
        x = self.dropout(self.input_nn(aa_indices, mod_x))
        charges_rep = charges.unsqueeze(1).repeat(1, x.size(1), 2)
        x = torch.cat((x, charges_rep), 2)
        hidden_x = self.hidden_nn(x)
        self.attentions = hidden_x[1] if self._output_attentions else None
        x = self.dropout(hidden_x[0] + x * 0.2)
        return self.output_nn(x).squeeze(1)


class Model_CCS_LSTM(nn.Module):
    """LSTM-based CCS prediction model"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        hidden = 256
        self.ccs_encoder = Encoder_26AA_Mod_Charge_CNN_LSTM_AttnSum(hidden)
        self.ccs_decoder = Decoder_Linear(hidden + 1, 1)

    def forward(self, aa_indices, mod_x, charges):
        x = self.ccs_encoder(aa_indices, mod_x, charges)
        x = self.dropout(x)
        x = torch.cat((x, charges), 1)
        return self.ccs_decoder(x).squeeze(1)


class IntenAwareLoss(nn.Module):
    """Intensity-weighted loss for MS2 models"""
    def __init__(self, base_weight=0.2):
        super().__init__()
        self.w = base_weight
        return torch.mean((target + self.w) * torch.abs(target - pred))


# ============================================================================
# MODEL INTERFACE BASE CLASS
# ============================================================================

class ModelInterface:
    """Base class for all prediction models with training and inference capabilities"""

    def __init__(self, device: str = 'gpu', fixed_sequence_len: int = 0, min_pred_value: float = 0.0, **kwargs):
        self.model: nn.Module = None
        self.optimizer = None
        self.model_params: dict = {}
        self.compiled = False
        self.set_device(device)
        self._fixed_sequence_len = fixed_sequence_len
        self._min_pred_value = min_pred_value
        self._target_column_to_predict = 'prediction'
        self._target_column_to_train = 'target'
        self.predict_df = None
        self._predict_in_order = False
        self.history = {'epoch': [], 'lr': [], 'train_loss': [], 'test_loss': []}
        self.loss_func = nn.L1Loss() # Default loss
        self._mask_modloss = False # Default: no masking
        

    @property
    def fixed_sequence_len(self):
        return self._fixed_sequence_len

    @fixed_sequence_len.setter
    def fixed_sequence_len(self, seq_len: int):
        self._fixed_sequence_len = seq_len
        self.model_params['fixed_sequence_len'] = seq_len

    @property
    def target_column_to_predict(self):
        return self._target_column_to_predict

    @target_column_to_predict.setter
    def target_column_to_predict(self, column: str):
        self._target_column_to_predict = column

    @property
    def target_column_to_train(self):
        return self._target_column_to_train

    @target_column_to_train.setter
    def target_column_to_train(self, column: str):
        self._target_column_to_train = column

    @property
    def device(self):
        return self._device

    @property
    def device_type(self):
        return self._device_type

    def set_device(self, device_type: str = 'gpu', device_ids: list = []):
        self._device_ids = device_ids
        if device_type == 'get_available':
            self._device, self._device_type = get_available_device()
        else:
            self._device, self._device_type = get_device(device_type, device_ids)
        self._model_to_device()

    def _model_to_device(self):
        if self.model is None:
            return
        if self._device_type != 'cuda':
            self.model.to(self._device)
        else:
            if self._device_ids and len(self._device_ids) > 1:
                self.model = nn.DataParallel(self.model, self._device_ids)
            elif not self._device_ids and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model.to(self._device)

    def build(self, model_class: nn.Module, **kwargs):
        self.model = model_class(**kwargs)
        self.model_params.update(**kwargs)
        self._model_to_device()
        self._init_for_training()

    def compile_model(self):
        """Apply torch.compile if available and not yet compiled."""
        if not self.compiled and hasattr(torch, 'compile'):
            # Only compile if device is CPU (per plan) or if requested explicitly
            # Note: The plan said disable by default. Calls to this method will be guarded by --torch_compile check in ai.py
            print("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model)
                self.compiled = True
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

    def _init_for_training(self):
        self.loss_func = nn.L1Loss()
        
        # Initialize AMP Scaler
        # Default to False for exact numerical parity with original script
        self.use_amp = False
        self.scaler = None

    def _as_tensor(self, data: np.ndarray, dtype=torch.float32):
        return torch.tensor(data, dtype=dtype, device=self._device)

    def set_lr(self, lr: float):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    def _get_lr_schedule_with_warmup(self, warmup_epoch, epoch):
        if warmup_epoch > epoch:
            warmup_epoch = epoch // 2
        return get_cosine_schedule_with_warmup(self.optimizer, warmup_epoch, epoch)

    def _pre_parse_data(self, df):
        """Pre-calculate AA indices and mod IDs once for the whole dataset."""
        if 'sequence' in df.columns and '_aa_indices' not in df.columns:
            # Faster sequence to index conversion (A=65, so s-64 = indices 1-26)
            # Use ASCII encoding directly (A-Z only)
            df['_aa_indices'] = [
                (np.frombuffer(s.encode('ascii'), dtype=np.int8) - 64).astype(np.int8)
                for s in df.sequence.values
            ]
            
        if 'mods' in df.columns and '_mod_id_list' not in df.columns:
            # Pre-parse modifications
            # Using list comprehension is faster than Series.apply(split)
            mod_lists = [s.split(';') if s else [] for s in df.mods.values]
            df['_mod_id_list'] = [
                np.array([MOD_TO_ID.get(m, -1) for m in ml if m], dtype=np.int32)
                for ml in mod_lists
            ]
            
            site_lists = [s.split(';') if s else [] for s in df.mod_sites.values]
            df['_mod_site_list'] = [
                np.array([int(s) for s in sl if s], dtype=np.int32)
                for sl in site_lists
            ]

    def _get_26aa_indice_features(self, batch_df: pd.DataFrame):
        return self._as_tensor(get_batch_aa_indices(batch_df), dtype=torch.long)

    def _get_mod_features(self, batch_df: pd.DataFrame):
        if self._fixed_sequence_len < 0:
            batch_df = batch_df.copy()
            batch_df['nAA'] = batch_df.nAA.max()
        return self._as_tensor(get_batch_mod_feature(batch_df))

    def _get_targets_from_batch_df(self, batch_df: pd.DataFrame, **kwargs):
        return self._as_tensor(batch_df[self._target_column_to_train].values, dtype=torch.float32)

    def _get_features_from_batch_df(self, batch_df: pd.DataFrame, **kwargs):
        return (self._get_26aa_indice_features(batch_df), self._get_mod_features(batch_df))

    def _prepare_training(self, precursor_df: pd.DataFrame, lr: float, **kwargs):
        if 'nAA' not in precursor_df.columns:
            precursor_df['nAA'] = precursor_df.sequence.str.len()
        
        # Optimize: Pre-parse strings once
        self._pre_parse_data(precursor_df)
        
        self._prepare_train_data_df(precursor_df, **kwargs)
        self.model.train()
        self.set_lr(lr)
        self.history = {'epoch': [], 'lr': [], 'train_loss': [], 'test_loss': []}

    def _prepare_train_data_df(self, precursor_df: pd.DataFrame, **kwargs):
        pass

    def _prepare_predict_data_df(self, precursor_df: pd.DataFrame, **kwargs):
        precursor_df[self._target_column_to_predict] = 0.0
        
        # Optimize: Pre-parse strings once
        self._pre_parse_data(precursor_df)
        
        self.predict_df = precursor_df

    def _train_one_batch(self, targets, *features, valid_targets=None):
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda'):
                predicts = self.model(*features)
                if valid_targets is not None:
                    mask = torch.where(valid_targets <= 0, 1.0, 0.0)
                    num_valid = torch.sum(mask)
                    masked_predicts = mask * predicts
                    masked_targets = mask * targets
                    
                    if isinstance(self.loss_func, nn.L1Loss):
                        if num_valid > 0:
                            cost = nn.L1Loss(reduction='sum')(masked_predicts, masked_targets) / num_valid
                        else:
                            cost = torch.tensor(0.0, device=self._device, requires_grad=True)
                    else:
                        cost = self.loss_func(masked_predicts, masked_targets)
                else:
                    cost = self.loss_func(predicts, targets)
            
            self.scaler.scale(cost).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predicts = self.model(*features)
            if valid_targets is not None:
                if True: # Modified to match peptdeep behavior: always mask if valid_targets exists
                    mask = torch.where(valid_targets <= 0, 1.0, 0.0)
                    num_valid = torch.sum(mask)
                    masked_predicts = mask * predicts
                    masked_targets = mask * targets
                    
                    if isinstance(self.loss_func, nn.L1Loss):
                        if num_valid > 0:
                            cost = nn.L1Loss(reduction='sum')(masked_predicts, masked_targets) / num_valid
                        else:
                            cost = torch.tensor(0.0, device=self._device, requires_grad=True)
                    else:
                        cost = self.loss_func(masked_predicts, masked_targets)
            else:
                cost = self.loss_func(predicts, targets)
            
            cost.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        return cost.item()

    def _train_one_epoch(self, precursor_df, epoch, batch_size, verbose_each_epoch, **kwargs):
        batch_cost = []
        _grouped = list(precursor_df.sample(frac=1).groupby('nAA'))
        rnd_nAA = np.random.permutation(len(_grouped))
        batch_tqdm = tqdm(rnd_nAA) if verbose_each_epoch else rnd_nAA

        for i_group in batch_tqdm:
            nAA, df_group = _grouped[i_group]
            
            # OPTIMIZATION: Prepare ENTIRE group features and targets once
            group_features = self._get_features_from_batch_df(df_group, **kwargs)
            group_targets = self._get_targets_from_batch_df(df_group, **kwargs)
            
            if kwargs.get("fragment_intensity_df_valid") is not None:
                group_valid_targets = self._get_valid_targets_from_batch_df(df_group, **kwargs)
            else:
                group_valid_targets = None

            for i in range(0, len(df_group), batch_size):
                # Slice Group Tensors (extremely fast)
                targets = group_targets[i:i + batch_size]
                
                if isinstance(group_features, tuple):
                    batch_features = [f[i:i + batch_size] for f in group_features]
                else:
                    batch_features = [group_features[i:i + batch_size]]
                
                valid_targets = group_valid_targets[i:i + batch_size] if group_valid_targets is not None else None

                batch_cost.append(self._train_one_batch(targets, *batch_features, valid_targets=valid_targets))

            if verbose_each_epoch:
                batch_tqdm.set_description(f'Epoch={epoch + 1}, nAA={nAA}, loss={batch_cost[-1]}')
        return batch_cost

    def train_with_warmup(self, precursor_df: pd.DataFrame, *, batch_size=1024, epoch=10,
                          warmup_epoch=5, lr=1e-4, verbose=False, verbose_each_epoch=False, **kwargs):
        self._prepare_training(precursor_df, lr, **kwargs)
        lr_scheduler = self._get_lr_schedule_with_warmup(warmup_epoch, epoch)

        for ep in range(epoch):
            batch_cost = self._train_one_epoch(precursor_df, ep, batch_size, verbose_each_epoch, **kwargs)
            lr_scheduler.step()
            if verbose:
                mask_str = " (Masked)" if kwargs.get("fragment_intensity_df_valid") is not None else ""
                print(f'[Training] Epoch={ep + 1}{mask_str}, lr={lr_scheduler.get_last_lr()[0]:.1e}, loss={np.mean(batch_cost)}')
            self.history['epoch'].append(ep + 1)
            self.history['lr'].append(lr_scheduler.get_last_lr()[0])
            self.history['train_loss'].append(np.mean(batch_cost))
            self.history['test_loss'].append(0) # No validation in this loop
        torch.cuda.empty_cache()

    def train_with_early_stopping(self, precursor_df: pd.DataFrame, *, batch_size=1024, epoch=20,
                                   warmup_epoch=5, lr=1e-4, val_df=None, patience=5, min_delta=0.001,
                                   checkpoint_path=None, verbose=True, verbose_each_epoch=False, **kwargs):
        """Train with early stopping and LR scheduling"""
        self._prepare_training(precursor_df, lr, **kwargs)
        lr_scheduler = self._get_lr_schedule_with_warmup(warmup_epoch, epoch)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for ep in range(epoch):
            batch_cost = self._train_one_epoch(precursor_df, ep, batch_size, verbose_each_epoch, **kwargs)
            train_loss = np.mean(batch_cost)
            lr_scheduler.step()

            if val_df is not None:
                val_loss = self._validate(val_df, batch_size, **kwargs)
                if verbose:
                    mask_str = " (Masked)" if kwargs.get("fragment_intensity_df_valid") is not None else ""
                    print(f'[Training] Epoch={ep + 1}{mask_str}, lr={lr_scheduler.get_last_lr()[0]:.1e}, '
                          f'train_loss={train_loss}, val_loss={val_loss}')
                
                self.history['epoch'].append(ep + 1)
                self.history['lr'].append(lr_scheduler.get_last_lr()[0])
                self.history['train_loss'].append(train_loss)
                self.history['test_loss'].append(val_loss)

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if checkpoint_path:
                        self.save(checkpoint_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {ep + 1}')
                    break
            else:
                if verbose:
                    print(f'[Training] Epoch={ep + 1}, lr={lr_scheduler.get_last_lr()[0]}, loss={train_loss}')
                self.history['epoch'].append(ep + 1)
                self.history['lr'].append(lr_scheduler.get_last_lr()[0])
                self.history['train_loss'].append(train_loss)
                self.history['test_loss'].append(0)

        torch.cuda.empty_cache()

    def _validate(self, val_df: pd.DataFrame, batch_size: int, **kwargs):
        """Compute validation loss"""
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for nAA, df_group in val_df.groupby('nAA'):
                for i in range(0, len(df_group), batch_size):
                    batch_df = df_group.iloc[i:i + batch_size, :]
                    targets = self._get_targets_from_batch_df(batch_df, **kwargs)
                    features = self._get_features_from_batch_df(batch_df, **kwargs)
                    if isinstance(features, tuple):
                        predicts = self.model(*features)
                    else:
                        predicts = self.model(features)
                    val_losses.append(self.loss_func(predicts, targets).item())
        self.model.train()
        return np.mean(val_losses)

    def train(self, precursor_df: pd.DataFrame, *, batch_size=1024, epoch=10, warmup_epoch=0,
              lr=1e-4, verbose=False, verbose_each_epoch=False, **kwargs):
        if warmup_epoch > 0:
            self.train_with_warmup(precursor_df, batch_size=batch_size, epoch=epoch,
                                    warmup_epoch=warmup_epoch, lr=lr, verbose=verbose,
                                    verbose_each_epoch=verbose_each_epoch, **kwargs)
        else:
            self._prepare_training(precursor_df, lr, **kwargs)
            for ep in range(epoch):
                batch_cost = self._train_one_epoch(precursor_df, ep, batch_size, verbose_each_epoch, **kwargs)
                if verbose:
                    print(f'[Training] Epoch={ep + 1}, Mean Loss={np.mean(batch_cost):.4f}')
                self.history['epoch'].append(ep + 1)
                self.history['lr'].append(lr) # Constant LR here
                self.history['train_loss'].append(np.mean(batch_cost))
                self.history['test_loss'].append(0)
            torch.cuda.empty_cache()

    def save_training_history(self, out_file):
        df = pd.DataFrame(self.history)
        df.to_csv(out_file, sep='\t', index=False)

    def plot_training_history(self, out_file, mark_best=True):
        import matplotlib.pyplot as plt
        df = pd.DataFrame(self.history)
        if len(df) == 0:
            return
        
        plt.figure(figsize=(6, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        
        # Only plot test loss if it has been recorded (sum > 0)
        if 'test_loss' in df.columns and df['test_loss'].sum() > 0:
             plt.plot(df['epoch'], df['test_loss'], label='Test Loss')
             
             if mark_best:
                 # Find best epoch among non-zero test losses
                 test_losses = df['test_loss'].values
                 valid_indices = np.where(test_losses > 0)[0]
                 if len(valid_indices) > 0:
                     best_valid_idx = np.argmin(test_losses[valid_indices])
                     best_idx = valid_indices[best_valid_idx]
                     best_epoch = df.iloc[best_idx]['epoch']
                     best_loss = df.iloc[best_idx]['test_loss']
                     plt.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7, 
                                label=f'Best Epoch ({int(best_epoch)})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()

    def _predict_one_batch(self, *features):
        return self.model(*features).cpu().detach().numpy()

    def predict(self, precursor_df: pd.DataFrame, *, batch_size=1024, verbose=False, **kwargs):
        if 'nAA' not in precursor_df.columns:
            precursor_df['nAA'] = precursor_df.sequence.str.len()
            precursor_df.sort_values('nAA', inplace=True)
            precursor_df.reset_index(drop=True, inplace=True)

        self._prepare_predict_data_df(precursor_df, **kwargs)
        self.model.eval()

        _grouped = precursor_df.groupby('nAA')
        batch_tqdm = tqdm(_grouped) if verbose else _grouped

        with torch.inference_mode():
            for nAA, df_group in batch_tqdm:
                # OPTIMIZATION: Featurize the ENTIRE group at once
                group_features = self._get_features_from_batch_df(df_group, **kwargs)
                group_size = len(df_group)
                group_results = []
                
                for i in range(0, group_size, batch_size):
                    # Slice the pre-calculated features
                    if isinstance(group_features, tuple):
                        batch_features = [f[i:i + batch_size] for f in group_features]
                    else:
                        batch_features = [group_features[i:i + batch_size]]
                    
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda'):
                            predicts = self._predict_one_batch(*batch_features)
                    else:
                        predicts = self._predict_one_batch(*batch_features)
                    
                    group_results.append(predicts)

                # Vectorized output assignment for the whole group
                if group_results:
                    self._set_batch_predict_data(df_group, np.concatenate(group_results), **kwargs)

        torch.cuda.empty_cache()
        return self.predict_df

    def _set_batch_predict_data(self, batch_df: pd.DataFrame, predict_values: np.ndarray, **kwargs):
        predict_values[predict_values < self._min_pred_value] = self._min_pred_value
        self.predict_df.loc[batch_df.index, self._target_column_to_predict] = predict_values

    def save(self, filename: str):
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.model.state_dict(), filename)
        logging.info(f"Saved model weights to {filename}")

    def load(self, model_file: Union[str, IO], model_name: str = None, **kwargs):
        """Load model weights from file, supporting zip and pt formats.
        
        Args:
            model_file: Path to .pt file or .zip file containing models
            model_name: When loading from zip, the name of the model file inside (e.g., 'ms2.pt')
        """
        if isinstance(model_file, str):
            if model_file.lower().endswith('.zip'):
                if model_name is None:
                    raise ValueError("model_name required when loading from zip file")
                with ZipFile(model_file, 'r') as zip_ref:
                    with zip_ref.open(model_name, 'r') as pt_file:
                        self._load_model_from_stream(pt_file)
            else:
                with open(model_file, 'rb') as pt_file:
                    self._load_model_from_stream(pt_file)
            logging.info(f"Loaded model weights from {model_file}")
        else:
            self._load_model_from_stream(model_file)
            logging.info("Loaded model weights from stream")

    def _load_model_from_stream(self, stream):
        missing_keys, unexpect_keys = self.model.load_state_dict(
            torch.load(stream, map_location=self._device), strict=False
        )
        # Filter out modloss keys - these are expected when loading with mask_modloss=True
        missing_keys = [k for k in missing_keys if not k.startswith('modloss')]
        unexpect_keys = [k for k in unexpect_keys if not k.startswith('modloss')]
        if len(missing_keys) > 0:
            logging.warning(f"nn parameters {missing_keys} are MISSING while loading models in {self.__class__}")
        if len(unexpect_keys) > 0:
            logging.warning(f"nn parameters {unexpect_keys} are UNEXPECTED while loading models in {self.__class__}")

    def get_parameter_num(self):
        return sum(p.numel() for p in self.model.parameters())

# ============================================================================
# HIGH-LEVEL MODEL CLASSES
# ============================================================================

class pDeepModel(ModelInterface):
    """MS2 prediction model interface"""
    def __init__(self, charged_frag_types=None, dropout=0.1, mask_modloss=True,
                 modloss_type='modloss', model_class=ModelMS2Bert, device='gpu', **kwargs):
        super().__init__(device=device)
        if charged_frag_types is None:
            charged_frag_types = get_charged_frag_types(FRAG_TYPES, MAX_FRAG_CHARGE)
        self.charged_frag_types = charged_frag_types
        self._modloss_frag_types = [i for i, frag in enumerate(charged_frag_types) if modloss_type in frag]

        self.charge_factor = 0.1
        self.NCE_factor = 0.01
        self.min_inten = 1e-4

        self.build(model_class, num_frag_types=len(charged_frag_types),
                   num_modloss_types=len(self._modloss_frag_types),
                   mask_modloss=mask_modloss, dropout=dropout, **kwargs)
        self._mask_modloss = mask_modloss
        # print("MS2 model initialized")

    def _prepare_train_data_df(self, precursor_df, fragment_intensity_df=None, fragment_intensity_df_valid=None, **kwargs):
        self.frag_inten_df = fragment_intensity_df[self.charged_frag_types] if fragment_intensity_df is not None else None
        self.frag_inten_df_valid = fragment_intensity_df_valid[self.charged_frag_types] if fragment_intensity_df_valid is not None else None

    def _prepare_predict_data_df(self, precursor_df, reference_frag_df=None, **kwargs):
        # Optimize: Pre-parse strings once
        self._pre_parse_data(precursor_df)
        
        self._predict_in_order = precursor_df.nAA.is_monotonic_increasing and reference_frag_df is None
        if self._predict_in_order and 'frag_start_idx' in precursor_df.columns:
            precursor_df.drop(columns=['frag_start_idx', 'frag_stop_idx'], inplace=True)
        self.predict_df = init_fragment_by_precursor_dataframe(
            precursor_df, self.charged_frag_types, reference_fragment_df=reference_frag_df, dtype=np.float32
        )

    def _get_features_from_batch_df(self, batch_df, **kwargs):
        aa_indices = self._get_26aa_indice_features(batch_df)
        mod_x = self._get_mod_features(batch_df)
        charges = self._as_tensor(batch_df['charge'].values).unsqueeze(1) * self.charge_factor
        nces = self._as_tensor(batch_df['nce'].values).unsqueeze(1) * self.NCE_factor
        instrument_indices = self._as_tensor(parse_instrument_indices(batch_df['instrument']), dtype=torch.long)
        return aa_indices, mod_x, charges, nces, instrument_indices

    def _get_targets_from_batch_df(self, batch_df, fragment_intensity_df=None, **kwargs):
        return self._as_tensor(
            get_sliced_fragment_dataframe(fragment_intensity_df, batch_df[['frag_start_idx', 'frag_stop_idx']].values).values
        ).view(-1, batch_df.nAA.values[0] - 1, len(self.charged_frag_types))

    def _get_valid_targets_from_batch_df(self, batch_df, fragment_intensity_df_valid=None, **kwargs):
        return self._as_tensor(
            get_sliced_fragment_dataframe(fragment_intensity_df_valid, batch_df[['frag_start_idx', 'frag_stop_idx']].values).values
        ).view(-1, batch_df.nAA.values[0] - 1, len(self.charged_frag_types))

    def _set_batch_predict_data(self, batch_df, predicts, **kwargs):
        apex_intens = predicts.reshape((len(batch_df), -1)).max(axis=1)
        apex_intens[apex_intens <= 0] = 1
        predicts /= apex_intens.reshape((-1, 1, 1))
        predicts[predicts < self.min_inten] = 0.0
        if self._predict_in_order:
            self.predict_df.values[batch_df.frag_start_idx.values[0]:batch_df.frag_stop_idx.values[-1], :] = \
                predicts.reshape((-1, len(self.charged_frag_types)))
        else:
            update_sliced_fragment_dataframe(self.predict_df, predicts.reshape((-1, len(self.charged_frag_types))),
                                              batch_df[['frag_start_idx', 'frag_stop_idx']].values)

    def train(self, precursor_df, fragment_intensity_df, fragment_intensity_df_valid=None, *,
              batch_size=1024, epoch=20, warmup_epoch=0, lr=1e-5, verbose=False, verbose_each_epoch=False, **kwargs):
        return super().train(precursor_df, fragment_intensity_df=fragment_intensity_df,
                              fragment_intensity_df_valid=fragment_intensity_df_valid, batch_size=batch_size,
                              epoch=epoch, warmup_epoch=warmup_epoch, lr=lr, verbose=verbose,
                              verbose_each_epoch=verbose_each_epoch, **kwargs)

    def grid_nce_search(self,
        psm_df:pd.DataFrame, 
        fragment_intensity_df:pd.DataFrame,
        nce_first=15, nce_last=45, nce_step=3,
        search_instruments = ['Lumos'],
        charged_frag_types:List = None,
        metric = 'PCC>0.9', # or 'median PCC'
        max_psm_subset = 1000000,
        callback = None
    ):
        print('Searching NCE...')
        print(f'NCE range: {nce_first} - {nce_last}, step: {nce_step}')
        print(f'Instruments: {search_instruments}')
        if len(psm_df) > max_psm_subset:
            psm_df = psm_df.sample(max_psm_subset).copy()
        best_pcc = -1
        best_nce = 0.
        best_instrument = None
        if 'median' in metric:
            metric_row = '50%'
        else:
            metric_row = '>0.90'
        
        # Resolve instrument groups from settings
        resolved_instruments = set()
        for inst in search_instruments:
            if inst in settings['model_mgr']['instrument_group']:
                 resolved_instruments.add(settings['model_mgr']['instrument_group'][inst])
            else:
                 resolved_instruments.add(inst)
        search_instruments = resolved_instruments

        for inst in search_instruments:
            for nce in np.arange(nce_first, nce_last+nce_step, nce_step):
                psm_df['nce'] = nce
                psm_df['instrument'] = inst
                predict_inten_df = self.predict(
                    psm_df, 
                    reference_frag_df=fragment_intensity_df
                )
                df, metrics = calc_ms2_similarity(
                    psm_df,
                    predict_inten_df, 
                    fragment_intensity_df,
                    charged_frag_types=charged_frag_types,
                    metrics=['PCC']
                )
                pcc = metrics.loc[metric_row, 'PCC']
                if pcc > best_pcc:
                    best_pcc = pcc
                    best_nce = nce
                    best_instrument = inst
        return best_nce, best_instrument

    def predict(self, precursor_df, *, batch_size=512, verbose=False, reference_frag_df=None, **kwargs):
        return super().predict(precursor_df, batch_size=batch_size, verbose=verbose,
                                reference_frag_df=reference_frag_df, **kwargs)


class AlphaRTModel(ModelInterface):
    """RT prediction model interface"""
    def __init__(self, dropout=0.1, model_class=Model_RT_LSTM_CNN, device='gpu', **kwargs):
        super().__init__(device=device)
        self.build(model_class, dropout=dropout, **kwargs)
        self.target_column_to_predict = 'rt_pred'
        self.target_column_to_train = 'rt_norm'

    def _get_features_from_batch_df(self, batch_df, **kwargs):
        return (self._get_26aa_indice_features(batch_df), self._get_mod_features(batch_df))

    def add_irt_column_to_precursor_df(self, df):
        """Add irt_pred column based on rt_pred using iRT peptides for calibration."""
        if 'rt_pred' not in df.columns:
            return df
            
        # Standard iRT peptides from Biognosys/PeptDeep
        irt_peptides = [
            ['LGGNEQVTR', 'RT-pep a', -24.92, '', ''],
            ['GAGSSEPVTGLDAK', 'RT-pep b', 0.00, '', ''],
            ['VEATFGVDESNAK', 'RT-pep c', 12.39, '', ''],
            ['YILAGVENSK', 'RT-pep d', 19.79, '', ''],
            ['TPVISGGPYEYR', 'RT-pep e', 28.71, '', ''],
            ['TPVITGAPYEYR', 'RT-pep f', 33.38, '', ''],
            ['DGLDAASYYAPVR', 'RT-pep g', 42.26, '', ''],
            ['ADVTPADFSEWSK', 'RT-pep h', 54.62, '', ''],
            ['GTFIIDPGGVIR', 'RT-pep i', 70.52, '', ''],
            ['GTFIIDPAAVIR', 'RT-pep k', 87.23, '', ''],
            ['LFLQFGAQGSPFLK', 'RT-pep l', 100.00, '', '']
        ]
        irt_df = pd.DataFrame(irt_peptides, columns=['sequence','pep_name','irt', 'mods', 'mod_sites'])
        irt_df['nAA'] = irt_df.sequence.str.len()
        
        # Predict RT for reference peptides
        # Use verbose=False to keep it quiet
        self.predict(irt_df, verbose=False)
        
        # Simple linear regression
        rt_pred_mean = irt_df.rt_pred.mean()
        irt_mean = irt_df.irt.mean()
        x = irt_df.rt_pred.values - rt_pred_mean
        y = irt_df.irt.values - irt_mean
        slope = np.sum(x*y)/np.sum(x*x)
        intercept = irt_mean - slope*rt_pred_mean
        
        df['irt_pred'] = df['rt_pred'] * slope + intercept
        return df


class AlphaCCSModel(ModelInterface):
    """CCS prediction model interface"""
    def __init__(self, dropout=0.1, model_class=Model_CCS_LSTM, device='gpu', **kwargs):
        super().__init__(device=device)
        self.build(model_class, dropout=dropout, **kwargs)
        self.charge_factor = 0.1
        self.target_column_to_predict = 'ccs_pred'
        self.target_column_to_train = 'ccs'

    def _get_features_from_batch_df(self, batch_df, **kwargs):
        aa_indices = self._get_26aa_indice_features(batch_df)
        mod_x = self._get_mod_features(batch_df)
        charges = self._as_tensor(batch_df['charge'].values).unsqueeze(1) * self.charge_factor
        return aa_indices, mod_x, charges

    def ccs_to_mobility_pred(self, precursor_df):
        return ccs_to_mobility_for_df(precursor_df, 'ccs_pred')


class ModelManager:
    """Manager class for MS2/RT/CCS models with training and prediction"""
    def __init__(self, mask_modloss=False, device='gpu', out_dir='./'):
        self.out_dir = out_dir
        self._device = device
        self.mask_modloss = mask_modloss
        self.nce = 30.0
        self._instrument = 'Lumos'
        self.charged_frag_types = get_charged_frag_types(FRAG_TYPES, MAX_FRAG_CHARGE)

        # Initialize models
        self.ms2_model = pDeepModel(charged_frag_types=self.charged_frag_types,
                                     mask_modloss=mask_modloss, device=device)
        self.rt_model = AlphaRTModel(device=device)
        self.ccs_model = AlphaCCSModel(device=device)

    @property
    def instrument(self):
        return self._instrument
    
    @instrument.setter
    def instrument(self, instrument_name: str):
        """Set instrument with mapping (e.g., Astral -> Lumos)"""
        instrument_name = instrument_name.upper()
        # Get instrument_group from peptdeep settings
        try:
            instrument_group = settings['model_mgr']['instrument_group']
            if instrument_name in instrument_group:
                self._instrument = instrument_group[instrument_name]
            else:
                self._instrument = 'Lumos'
        except (KeyError, TypeError):
            self._instrument = 'Lumos'


    @property
    def device(self):
        return self._device

    def set_device(self, device):
        self._device = device
        self.ms2_model.set_device(device)
        self.rt_model.set_device(device)
        self.ccs_model.set_device(device)

    def load_installed_models(self, model_type='generic', model_list=None):
        """Load pretrained models from peptdeep installation.
        
        Args:
            model_type: 'generic' or 'phos' for phosphorylation models
            model_list: List of model types to load individually (e.g. ['rt'] or ['ms2']).
                        If None, loads all available models (ms2, rt, ccs).
                        Valid values: 'ms2', 'rt', 'ccs'.
        """
        # Try to find the peptdeep pretrained models zip
        model_zip = None
        
        # Check common locations
        peptdeep_home = global_settings.get('PEPTDEEP_HOME', os.path.expanduser('~/peptdeep'))
        possible_paths = [
            os.path.join(peptdeep_home, 'pretrained_models', 'pretrained_models.zip'),
            '/data/peptdeep/pretrained_models/pretrained_models.zip',
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                model_zip = path
                break
        
        if model_zip is None:
            print("Pretrained models not found. Attempting download...")
            model_zip = download_models()
        
        if model_zip is None:
            print("Warning: No pretrained models found. Training from scratch.")
            return
        
        print(f"Loading pretrained models from: {model_zip}")
        
        # Model file names inside the zip depend on model_type
        if model_type == 'phos' or model_type == 'phospho' or model_type == 'phosphorylation':
            ms2_name = 'generic/ms2.pth'  # match peptdeep: use generic MS2 for phos
            rt_name = 'phospho/rt_phos.pth'
            ccs_name = 'generic/ccs.pth'  # match peptdeep: use generic CCS for phos
        elif model_type == 'generic':
            ms2_name = 'generic/ms2.pth'
            rt_name = 'generic/rt.pth'
            ccs_name = 'generic/ccs.pth'
        elif model_type == 'kGG' or model_type == 'kgg' or model_type == 'ubiquitination' or model_type == 'ubi':
            ms2_name = 'generic/ms2.pth'
            rt_name = 'digly/rt_digly.pth'
            ccs_name = 'generic/ccs.pth'
        else:
            print(f"Warning: Unknown model type '{model_type}'. Loading generic models.")   
            ms2_name = 'generic/ms2.pth'
            rt_name = 'generic/rt.pth'
            ccs_name = 'generic/ccs.pth'
        
        # Determine strict list of models to load
        if model_list is None:
             model_list = ['ms2', 'rt', 'ccs']
        
        self.load_external_models(
            ms2_model_file=model_zip if 'ms2' in model_list else '',
            ms2_model_name=ms2_name,
            rt_model_file=model_zip if 'rt' in model_list else '',
            rt_model_name=rt_name,
            ccs_model_file=model_zip if 'ccs' in model_list else '',
            ccs_model_name=ccs_name
        )

    def load_external_models(self, *, ms2_model_file='', ms2_model_name='ms2.pt',
                             rt_model_file='', rt_model_name='rt.pt',
                             ccs_model_file='', ccs_model_name='ccs.pt'):
        """Load external model files.
        
        Args:
            ms2_model_file: Path to MS2 model file or zip
            ms2_model_name: Name of MS2 model inside zip (if zip file)
            rt_model_file: Path to RT model file or zip
            rt_model_name: Name of RT model inside zip (if zip file)
            ccs_model_file: Path to CCS model file or zip
            ccs_model_name: Name of CCS model inside zip (if zip file)
        """
        try:
            if ms2_model_file:
                self.ms2_model.load(ms2_model_file, model_name=ms2_model_name)
                print(f"Loaded MS2 model from {ms2_model_file}")
        except Exception as e:
            print(f"Warning: Could not load MS2 model: {e}")
        
        try:
            if rt_model_file:
                self.rt_model.load(rt_model_file, model_name=rt_model_name)
                print(f"Loaded RT model from {rt_model_file}")
        except Exception as e:
            print(f"Warning: Could not load RT model: {e}")
        
        try:
            if ccs_model_file:
                self.ccs_model.load(ccs_model_file, model_name=ccs_model_name)
                print(f"Loaded CCS model from {ccs_model_file}")
        except Exception as e:
            print(f"Warning: Could not load CCS model: {e}")

    def save_models(self, folder):
        """Save all models to folder"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.ms2_model.save(os.path.join(folder, 'ms2.pt'))
        self.rt_model.save(os.path.join(folder, 'rt.pt'))
        self.ccs_model.save(os.path.join(folder, 'ccs.pt'))
        print(f"Models saved to {folder}")

    def set_default_nce_instrument(self, df):
        """Add nce and instrument columns to dataframe"""
        df['nce'] = self.nce
        df['instrument'] = self.instrument
        return df

    def train_ms2_model(self, psm_df, fragment_intensity_df, *, batch_size=1024, epoch=20,
                         warmup_epoch=5, lr=1e-4, val_df=None, patience=5, early_stopping=True,
                         verbose=True, torch_compile=False, **kwargs):
        """Train MS2 model with optional early stopping"""
        if torch_compile:
            self.ms2_model.compile_model()

        if early_stopping and val_df is not None:
            self.ms2_model.train_with_early_stopping(
                psm_df, fragment_intensity_df=fragment_intensity_df,
                batch_size=batch_size, epoch=epoch, warmup_epoch=warmup_epoch, lr=lr,
                val_df=val_df, patience=patience, verbose=verbose, **kwargs
            )
        else:
            self.ms2_model.train(
                psm_df, fragment_intensity_df=fragment_intensity_df,
                batch_size=batch_size, epoch=epoch, warmup_epoch=warmup_epoch, lr=lr,
                verbose=verbose, **kwargs
            )

    def train_rt_model(self, psm_df, *, batch_size=1024, epoch=40, warmup_epoch=5, lr=1e-4,
                        val_df=None, patience=5, early_stopping=True, verbose=True, torch_compile=False, **kwargs):
        """Train RT model with optional early stopping"""
        if torch_compile:
            self.rt_model.compile_model()

        if early_stopping and val_df is not None:
            self.rt_model.train_with_early_stopping(
                psm_df, batch_size=batch_size, epoch=epoch, warmup_epoch=warmup_epoch, lr=lr,
                val_df=val_df, patience=patience, verbose=verbose, **kwargs
            )
        else:
            self.rt_model.train(psm_df, batch_size=batch_size, epoch=epoch,
                                 warmup_epoch=warmup_epoch, lr=lr, verbose=verbose, **kwargs)

    def train_ccs_model(self, psm_df, *, batch_size=1024, epoch=40, warmup_epoch=5, lr=1e-4,
                         val_df=None, patience=5, early_stopping=True, verbose=True, torch_compile=False, **kwargs):
        """Train CCS model with optional early stopping"""
        if torch_compile:
            self.ccs_model.compile_model()

        if early_stopping and val_df is not None:
            self.ccs_model.train_with_early_stopping(
                psm_df, batch_size=batch_size, epoch=epoch, warmup_epoch=warmup_epoch, lr=lr,
                val_df=val_df, patience=patience, verbose=verbose, **kwargs
            )
        else:
            self.ccs_model.train(psm_df, batch_size=batch_size, epoch=epoch,
                                  warmup_epoch=warmup_epoch, lr=lr, verbose=verbose, **kwargs)

    def predict_ms2(self, df, batch_size=512, **kwargs):
        """Predict MS2 intensities and return intensity dataframe"""
        if 'instrument' not in df.columns:
            df['instrument'] = self.instrument
        if 'nce' not in df.columns:
            df['nce'] = self.nce
        # Check for torch_compile in kwargs
        if kwargs.get('torch_compile', False):
            self.ms2_model.compile_model()
        return self.ms2_model.predict(df, batch_size=batch_size, **kwargs)

    def predict_rt(self, df, batch_size=1024, **kwargs):
        """Predict RT and return dataframe with rt_pred column"""
        # Check for torch_compile in kwargs
        if kwargs.get('torch_compile', False):
            self.rt_model.compile_model()
        df = self.rt_model.predict(df, batch_size=batch_size, **kwargs)
        df['rt_norm_pred'] = df['rt_pred']
        return df

    def predict_mobility(self, df, batch_size=1024, **kwargs):
        """Predict CCS/mobility and return dataframe with mobility_pred column"""
        # Check for torch_compile in kwargs
        if kwargs.get('torch_compile', False):
            self.ccs_model.compile_model()
        self.ccs_model.predict(df, batch_size=batch_size, **kwargs)
        df['mobility_pred'] = self.ccs_model.ccs_to_mobility_pred(df)
        return df
