import streamlit as st
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.plotting import plot_stat_map
from nilearn import image
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import re
import os
import gdown  # <--- NEW IMPORT
from pathlib import Path
import shutil  # <--- NEW IMPORT
import zipfile  # <--- Add this to your imports at the top

st.set_page_config(layout="wide", page_title="Compare Activation Maps")

# ======================= CLOUD DATA SETUP =======================

# 1. SETUP LOCAL PATH
DATA_ROOT = Path("downloaded_data")

# 2. YOUR *ZIP FILE* ID (NOT the folder ID)
# Paste the ID of the 'fmri_data.zip' file here
ZIP_FILE_ID = "1u61zPpdCjlPvTuwttGkCkPIOLXUFcnXS"
#https://drive.google.com/file/d/1u61zPpdCjlPvTuwttGkCkPIOLXUFcnXS/view?usp=sharing

@st.cache_resource
def download_data_folder():
    """Downloads and unzips data from Drive once at startup"""
    if DATA_ROOT.exists():
        # Check if not empty
        if any(DATA_ROOT.iterdir()):
            return True

    st.warning("⏳ Downloading data... (This happens once)")

    try:
        # 1. Download the ZIP file
        zip_path = "temp_data.zip"
        gdown.download(id=ZIP_FILE_ID, output=zip_path, quiet=False)

        # 2. Unzip it
        st.info("📦 Unzipping data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_ROOT)

        # 3. Clean up
        os.remove(zip_path)

        st.success("✅ Data ready!")
        return True

    except Exception as e:
        st.error(f"❌ Error: {e}")
        # Cleanup if failed
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if DATA_ROOT.exists():
            shutil.rmtree(DATA_ROOT)
        st.stop()


# Trigger download
download_data_folder()



# ======================= CONFIGURATION =======================

# 3. UPDATE PATHS TO POINT TO THE DOWNLOADED FOLDER
# The names "fmriprep", "spm", "confounds" must match your Drive folder names

DATA_SOURCES = {
    "fMRIPrep GLM": {
        "path": str(DATA_ROOT / "fmriprep"),
        "enabled": True,
        "color": "#1f77b4",
    },
    "SPM GLM": {
        "path": str(DATA_ROOT / "spm"),
        "enabled": True,
        "color": "#ff7f0e",
    },
}

CONFOUNDS_ROOT = str(DATA_ROOT / "confounds")
BEHAVIORAL_DATA_PATH = DATA_ROOT / "behavioral" / "DFG_behavioural_results.csv"

# ... REST OF YOUR CODE (Keep everything else exactly the same) ...


# Condition mapping
CONDITION_LABELS = [
    "condition 1- audio and visual- same",
    "condition 2-audio only",
    "condition 3-visual only",
    "condition 4-audio and visual - different",
]

# Also accept short names and map them to full names
CONDITION_MAPPING = {
    "condition 1- audio and visual- same": 0,
    "condition 2-audio only": 1,
    "condition 3-visual only": 2,
    "condition 4-audio and visual - different": 3,
    # Short name mappings
    "AV_Same": 0,
    "Audio_only": 1,
    "Visual_only": 2,
    "AV_Diff": 3,
}

# Default cognitive features to display
DEFAULT_COG_FEATURES = [
    "NamingMixObj_Aleftaf_RawPerMin",
    "NamingLetters_Shatil_Raw",
    "NamingLetters_Shatil_Time",
    "SkySearch_TEACH_Attention_Score",
    "Age_at_Scan",
]

st.title("🔬 Activation Map Comparison Tool")
st.markdown("Compare activation maps from multiple GLM sources")

# ======================= DATA LOADING FUNCTIONS =======================

@st.cache_data
def load_behavioral_data():
    """Load behavioral data CSV."""
    if not BEHAVIORAL_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(BEHAVIORAL_DATA_PATH)
        return df
    except Exception as e:
        st.warning(f"Could not load behavioral data: {e}")
        return None

@st.cache_data
def discover_all_maps(source_paths: Dict[str, str]) -> Dict[str, List[Dict]]:
    """
    Discover all available maps from all enabled sources.
    
    Returns:
        Dict mapping source_name -> list of map metadata dicts
    """
    all_maps = {}
    
    for source_name, source_config in source_paths.items():
        if not source_config["enabled"]:
            continue
            
        source_path = Path(source_config["path"])
        if not source_path.exists():
            st.warning(f"Source path not found: {source_path}")
            continue
        
        maps = []
        
        # Find all subject folders
        for subj_folder in sorted(source_path.glob("subject_*")):
            if not subj_folder.is_dir():
                continue
            
            subj_id = int(subj_folder.name.split("_")[1])
            
            # Find all session folders
            for sess_folder in sorted(subj_folder.glob("session_*")):
                if not sess_folder.is_dir():
                    continue
                
                sess_id = int(sess_folder.name.split("_")[1])
                
                # Find all nifti files
                for nii_file in sorted(sess_folder.glob("*.nii*")):
                    # Parse filename to extract metadata
                    filename = nii_file.name
                    
                    # Check if it's a condition map or contrast map
                    is_contrast = "contrast" in filename.lower()
                    
                    if is_contrast:
                        # Extract contrast name
                        # Pattern 1: *_contrast-XXX_zmap.nii.gz (fMRIPrep)
                        match = re.search(r'contrast-(.+?)_zmap', filename)
                        if not match:
                            # Pattern 2: zmap_contrast_XXX.nii (SPM)
                            match = re.search(r'zmap_contrast_(.+?)\.nii', filename)
                        
                        if match:
                            contrast_name = match.group(1)
                            
                            # Normalize contrast name: replace underscores with spaces and clean up
                            # SPM: "condition_1-_audio_and_visual-_same_vs_condition_2-audio_only"
                            # Should become: "condition 1- audio and visual- same vs condition 2-audio only"
                            contrast_name = contrast_name.replace('_', ' ')
                            # Fix double spaces
                            contrast_name = re.sub(r'\s+', ' ', contrast_name)
                            # Fix "condition X-" pattern (remove space before dash)
                            contrast_name = re.sub(r'condition (\d+)- ', r'condition \1- ', contrast_name)
                            
                            maps.append({
                                'source': source_name,
                                'subject_id': subj_id,
                                'session_id': sess_id,
                                'type': 'contrast',
                                'contrast_name': contrast_name,
                                'file_path': str(nii_file),
                                'filename': filename,
                            })
                    else:
                        # Extract condition - try both patterns
                        # Pattern 1: *_cond-XXX_zmap.nii.gz
                        match = re.search(r'cond[-_](.+?)_zmap', filename)
                        if not match:
                            # Pattern 2: zmap_XX_condition_X-*.nii (SPM format)
                            match = re.search(r'condition[_\s](\d+)', filename)
                            if match:
                                cond_num = int(match.group(1))
                                condition_name = CONDITION_LABELS[cond_num - 1] if cond_num <= len(CONDITION_LABELS) else f"condition_{cond_num}"
                                cond_id = cond_num - 1
                        else:
                            condition_name = match.group(1)
                            # Map to condition ID (handle both short and long names)
                            cond_id = CONDITION_MAPPING.get(condition_name, -1)
                            # If it's a short name, convert to long name
                            if cond_id >= 0 and cond_id < len(CONDITION_LABELS):
                                condition_name = CONDITION_LABELS[cond_id]
                        
                        if match and cond_id >= 0:
                            maps.append({
                                'source': source_name,
                                'subject_id': subj_id,
                                'session_id': sess_id,
                                'type': 'condition',
                                'condition_id': cond_id,
                                'condition_name': condition_name,
                                'file_path': str(nii_file),
                                'filename': filename,
                            })
        
        all_maps[source_name] = maps
    
    return all_maps

@st.cache_data
def load_fd_metrics(subject_id: int, session_id: int) -> Optional[Dict]:
    """
    Load FD (framewise displacement) metrics for a subject/session.
    
    Returns:
        Dict with 'mean_fd', 'bad_volumes', 'total_volumes', 'percent_bad'
        or None if not available
    """
    confounds_root = Path(CONFOUNDS_ROOT)
    if not confounds_root.exists():
        return None
    
    # Find subject folder
    subj_folder = confounds_root / f"sub-{subject_id}"
    if not subj_folder.exists():
        return None
    
    # Find session folder
    sess_folder = subj_folder / f"ses-{session_id}" / "func"
    if not sess_folder.exists():
        return None
    
    # Find all confounds files for this session
    # Pattern may or may not include ses-X in filename
    confounds_files = list(sess_folder.glob(f"*_desc-confounds_timeseries.tsv"))
    
    # Filter to only include files for this session (if ses-X is in filename)
    # or accept all files if no session info in filename
    filtered_files = []
    for f in confounds_files:
        # Check if filename contains session info
        if f"_ses-{session_id}_" in f.name or "_ses-" not in f.name:
            filtered_files.append(f)
    
    confounds_files = filtered_files
    
    if len(confounds_files) == 0:
        return None
    
    # Aggregate FD across all runs
    all_fd = []
    fd_threshold = 0.5
    
    for conf_file in confounds_files:
        try:
            df = pd.read_csv(conf_file, sep='\t')
            if 'framewise_displacement' in df.columns:
                fd = df['framewise_displacement'].values
                all_fd.append(fd)
        except Exception as e:
            continue
    
    if len(all_fd) == 0:
        return None
    
    # Concatenate all FD values
    all_fd_concat = np.concatenate(all_fd)
    
    # Calculate metrics
    mean_fd = np.nanmean(all_fd_concat)
    bad_volumes = np.sum(all_fd_concat > fd_threshold)
    total_volumes = len(all_fd_concat)
    percent_bad = (bad_volumes / total_volumes) * 100 if total_volumes > 0 else 0
    
    return {
        'mean_fd': mean_fd,
        'bad_volumes': bad_volumes,
        'total_volumes': total_volumes,
        'percent_bad': percent_bad,
        'n_runs': len(confounds_files),
    }

@st.cache_data
def get_behavioral_values(behavioral_df: pd.DataFrame, subject_id: int, session_id: int) -> Optional[Dict]:
    """
    Extract behavioral/cognitive values for a subject and session.
    
    Returns:
        Dict with behavioral metrics or None if not available
    """
    if behavioral_df is None:
        return None
    
    # Filter for this subject
    subj_data = behavioral_df[behavioral_df['Subject'] == subject_id]
    
    if len(subj_data) == 0:
        return None
    
    # The behavioral CSV has columns with #1, #2, #3 suffixes for different sessions
    session_suffix = f"#{session_id}"
    
    # Extract relevant columns for this session
    result = {}
    
    # Use only default cognitive features
    for metric_base in DEFAULT_COG_FEATURES:
        col_name = f"{metric_base}{session_suffix}"
        if col_name in subj_data.columns:
            values = subj_data[col_name].dropna()
            if len(values) > 0 and values.iloc[0] != '':
                try:
                    val = float(values.iloc[0])
                    # Use a cleaner name (remove session suffix)
                    clean_name = metric_base.replace('_SS', '').replace('_', ' ')
                    result[clean_name] = val
                except (ValueError, TypeError):
                    continue
    
    return result if result else None

@st.cache_data
def load_nifti_image(file_path: str):
    """Load a NIfTI image."""
    try:
        return image.load_img(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

@st.cache_data
def create_dummy_masker():
    """Create a simple masker for visualization (uses first available image)."""
    # Try to find any nifti file to extract shape/affine
    for source_name, source_config in DATA_SOURCES.items():
        if not source_config["enabled"]:
            continue
        source_path = Path(source_config["path"])
        if not source_path.exists():
            continue
        
        # Find first nifti file
        nii_files = list(source_path.glob("**/subject_*/session_*/*.nii*"))
        if nii_files:
            try:
                img = image.load_img(nii_files[0])
                # Create a simple brain mask (non-zero voxels)
                data = img.get_fdata()
                mask = np.abs(data) > 1e-6
                mask_img = nib.Nifti1Image(mask.astype(np.int32), img.affine)
                return NiftiMasker(mask_img=mask_img).fit()
            except:
                continue
    
    return None

# ======================= LOAD DATA =======================

# Load behavioral data
behavioral_df = load_behavioral_data()

# Discover all maps
with st.spinner("Discovering activation maps..."):
    all_maps = discover_all_maps(DATA_SOURCES)

# Create masker
masker = create_dummy_masker()

# Count maps per source
total_maps = sum(len(maps) for maps in all_maps.values())

if total_maps == 0:
    st.error("No activation maps found in any source!")
    st.stop()

st.success(f"✅ Loaded {total_maps} maps from {len(all_maps)} sources")

# Show source summary
with st.expander("📊 Data Source Summary"):
    for source_name, maps in all_maps.items():
        n_condition = sum(1 for m in maps if m['type'] == 'condition')
        n_contrast = sum(1 for m in maps if m['type'] == 'contrast')
        st.write(f"**{source_name}**: {len(maps)} maps ({n_condition} conditions, {n_contrast} contrasts)")

# ======================= SIDEBAR: VIEW MODE SELECTION =======================

st.sidebar.header("🎯 View Mode")

view_mode = st.sidebar.radio(
    "Select viewing mode:",
    ["By Subject & Session", "By Contrast"],
    key="view_mode"
)

# ======================= SIDEBAR: SOURCE SELECTION =======================

st.sidebar.header("📂 Data Sources")

selected_sources = []
for source_name in all_maps.keys():
    if st.sidebar.checkbox(source_name, value=True, key=f"source_{source_name}"):
        selected_sources.append(source_name)

if len(selected_sources) == 0:
    st.warning("Please select at least one data source")
    st.stop()

# ======================= MODE 1: BY SUBJECT & SESSION =======================

if view_mode == "By Subject & Session":
    st.header("📍 View by Subject & Session")
    
    # Get all unique (subject, session) combinations
    all_combinations = set()
    for source_name in selected_sources:
        for map_info in all_maps[source_name]:
            all_combinations.add((map_info['subject_id'], map_info['session_id']))
    
    all_combinations = sorted(list(all_combinations))
    
    if len(all_combinations) == 0:
        st.warning("No maps found for selected sources")
        st.stop()
    
    # Navigation - by subject and session
    col1, col2, col3 = st.columns(3)
    
    # Get unique subjects and sessions
    all_subjects = sorted(list(set([s for s, _ in all_combinations])))
    all_sessions = sorted(list(set([ses for _, ses in all_combinations])))
    
    with col1:
        subj_id = st.selectbox(
            "Subject ID",
            all_subjects,
            key="selected_subject"
        )
    
    with col2:
        # Filter sessions available for this subject
        available_sessions = sorted([ses for s, ses in all_combinations if s == subj_id])
        sess_id = st.selectbox(
            "Session ID",
            available_sessions,
            key="selected_session"
        )
    
    with col3:
        # Show index position
        try:
            current_idx = all_combinations.index((subj_id, sess_id))
            st.metric("Position", f"{current_idx + 1} / {len(all_combinations)}")
        except ValueError:
            st.metric("Position", "N/A")
    
    st.markdown(f"## Subject {subj_id} | Session {sess_id}")
    
    # ======================= DISPLAY FD METRICS =======================
    
    fd_metrics = load_fd_metrics(subj_id, sess_id)
    
    if fd_metrics:
        st.markdown("### 📊 Motion Metrics (FD - Framewise Displacement)")
        fd_cols = st.columns(5)
        with fd_cols[0]:
            st.metric("Mean FD", f"{fd_metrics['mean_fd']:.3f} mm")
        with fd_cols[1]:
            st.metric("Bad Volumes", f"{fd_metrics['bad_volumes']}")
        with fd_cols[2]:
            st.metric("Total Volumes", f"{fd_metrics['total_volumes']}")
        with fd_cols[3]:
            st.metric("% Bad", f"{fd_metrics['percent_bad']:.1f}%")
        with fd_cols[4]:
            st.metric("# Runs", f"{fd_metrics['n_runs']}")
        
        # Color code quality
        if fd_metrics['mean_fd'] < 0.2:
            st.success("✅ Excellent motion quality")
        elif fd_metrics['mean_fd'] < 0.5:
            st.info("ℹ️ Good motion quality")
        elif fd_metrics['mean_fd'] < 0.9:
            st.warning("⚠️ Moderate motion")
        else:
            st.error("❌ High motion - consider excluding")
    else:
        st.info("Motion metrics not available")
    
    # ======================= DISPLAY BEHAVIORAL DATA =======================
    
    behav_values = get_behavioral_values(behavioral_df, subj_id, sess_id)
    
    if behav_values:
        st.markdown("### 🧠 Cognitive/Behavioral Values")
        behav_cols = st.columns(min(len(behav_values), 6))
        for idx, (metric_name, metric_val) in enumerate(behav_values.items()):
            with behav_cols[idx % 6]:
                st.metric(metric_name, f"{metric_val:.1f}")
    else:
        st.info("Behavioral data not available for this subject/session")
    
    # ======================= DISPLAY CONDITION MAPS =======================
    
    st.markdown("---")
    st.markdown("### 🗺️ Condition Activation Maps")
    
    # Plot controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        n_slices = st.number_input("# Slices", 0, 20, 7, key="n_slices_cond")
    with ctrl_col2:
        use_symmetric = st.checkbox("Symmetric colorbar", value=True, key="symmetric_cond")
    with ctrl_col3:
        vmax_cond = st.number_input("vmax", value=5.0, format="%.1f", key="vmax_cond")
    
    vmin_cond = -vmax_cond if use_symmetric else 0.0
    
    # Compute cut coordinates
    cut_coords = None
    if masker and n_slices > 0:
        try:
            mask_img = masker.mask_img_
            affine = mask_img.affine
            z_bounds = (affine[2, 3], affine[2, 3] + affine[2, 2] * mask_img.shape[2])
            cut_coords = np.linspace(z_bounds[0], z_bounds[1], n_slices + 2)[1:-1]
        except:
            cut_coords = None
    
    # For each condition, show maps from all sources stacked vertically
    for cond_id, cond_name in enumerate(CONDITION_LABELS):
        st.markdown(f"#### Condition {cond_id}: {cond_name}")
        
        # Show maps from each source, stacked vertically
        for source_name in selected_sources:
            st.markdown(f"**{source_name}**")
            
            # Find map
            matching_maps = [
                m for m in all_maps[source_name]
                if m['subject_id'] == subj_id
                and m['session_id'] == sess_id
                and m['type'] == 'condition'
                and m['condition_id'] == cond_id
            ]
            
            if len(matching_maps) == 0:
                st.warning(f"Map not found for {source_name}")
                continue
            
            map_info = matching_maps[0]
            img = load_nifti_image(map_info['file_path'])
            
            if img is None:
                st.error(f"Failed to load: {map_info['file_path']}")
                continue
            
            # Plot
            fig = plt.figure(figsize=(14, 4))
            try:
                plot_stat_map(
                    img,
                    bg_img=None,
                    display_mode="z",
                    cut_coords=cut_coords,
                    cmap="cold_hot",
                    vmin=vmin_cond,
                    vmax=vmax_cond,
                    title=f"{source_name} | Sub {subj_id} Ses {sess_id} | {cond_name}",
                    figure=fig,
                    symmetric_cbar=use_symmetric,
                    threshold=0.001,
                )
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Plotting error: {e}")
                plt.close(fig)
            
            # Stats
            data = img.get_fdata().flatten()
            data = data[~np.isnan(data) & (data != 0)]
            if len(data) > 0:
                st.caption(f"Mean: {data.mean():.2f} | Std: {data.std():.2f} | Max: {data.max():.2f}")
        
        st.markdown("---")
    
    # ======================= DISPLAY CONTRAST MAPS =======================
    
    st.markdown("---")
    st.markdown("### 🔀 Contrast Maps (Condition vs Condition)")
    
    # Find all available contrasts for this subject/session
    available_contrasts = set()
    for source_name in selected_sources:
        for map_info in all_maps[source_name]:
            if (map_info['subject_id'] == subj_id and 
                map_info['session_id'] == sess_id and 
                map_info['type'] == 'contrast'):
                available_contrasts.add(map_info['contrast_name'])
    
    if len(available_contrasts) > 0:
        for contrast_name in sorted(available_contrasts):
            st.markdown(f"#### Contrast: {contrast_name}")
            
            # Show maps from each source, stacked vertically
            for source_name in selected_sources:
                # Find map
                matching_maps = [
                    m for m in all_maps[source_name]
                    if m['subject_id'] == subj_id
                    and m['session_id'] == sess_id
                    and m['type'] == 'contrast'
                    and m['contrast_name'] == contrast_name
                ]
                
                if len(matching_maps) == 0:
                    continue
                
                st.markdown(f"**{source_name}**")
                map_info = matching_maps[0]
                img = load_nifti_image(map_info['file_path'])
                
                if img is None:
                    st.error(f"Failed to load: {map_info['file_path']}")
                    continue
                
                # Plot
                fig = plt.figure(figsize=(14, 4))
                try:
                    plot_stat_map(
                        img,
                        bg_img=None,
                        display_mode="z",
                        cut_coords=cut_coords,
                        cmap="cold_hot",
                        vmin=vmin_cond,
                        vmax=vmax_cond,
                        title=f"{source_name} | Sub {subj_id} Ses {sess_id} | {contrast_name}",
                        figure=fig,
                        symmetric_cbar=use_symmetric,
                        threshold=0.001,
                    )
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"Plotting error: {e}")
                    plt.close(fig)
                
                # Stats
                data = img.get_fdata().flatten()
                data = data[~np.isnan(data) & (data != 0)]
                if len(data) > 0:
                    st.caption(f"Mean: {data.mean():.2f} | Std: {data.std():.2f} | Max: {data.max():.2f}")
            
            st.markdown("---")
    else:
        st.info("No contrast maps available for this subject/session")

# ======================= MODE 2: BY CONDITION/CONTRAST =======================

elif view_mode == "By Contrast":
    st.header("🔀 View by Condition/Contrast")
    
    # Gather all unique conditions and contrasts
    all_conditions = []
    all_contrasts = set()
    
    for source_name in selected_sources:
        for map_info in all_maps[source_name]:
            if map_info['type'] == 'contrast':
                all_contrasts.add(map_info['contrast_name'])
    
    # Build selection list: conditions first, then contrasts
    selection_items = []
    for cond_id, cond_name in enumerate(CONDITION_LABELS):
        selection_items.append(("condition", cond_name, cond_id))
    
    for contrast_name in sorted(all_contrasts):
        selection_items.append(("contrast", contrast_name, None))
    
    if len(selection_items) == 0:
        st.warning("No conditions or contrasts found")
        st.stop()
    
    # Create display names for selection
    display_names = []
    for item_type, item_name, item_id in selection_items:
        if item_type == "condition":
            display_names.append(f"📊 {item_name}")
        else:
            display_names.append(f"🔀 {item_name}")
    
    # Select condition or contrast
    selected_idx = st.selectbox(
        "Select condition or contrast:",
        range(len(selection_items)),
        format_func=lambda x: display_names[x],
        key="selected_item"
    )
    
    item_type, item_name, item_id = selection_items[selected_idx]
    
    st.markdown(f"## {display_names[selected_idx]}")
    
    # Find all maps for this selection across all subjects/sessions/sources
    matching_maps = []
    for source_name in selected_sources:
        for map_info in all_maps[source_name]:
            if item_type == "condition":
                if map_info['type'] == 'condition' and map_info['condition_id'] == item_id:
                    matching_maps.append((source_name, map_info))
            else:  # contrast
                if map_info['type'] == 'contrast' and map_info['contrast_name'] == item_name:
                    matching_maps.append((source_name, map_info))
    
    # Sort maps: first by (subject, session), then by source
    # This groups maps from the same subject/session together
    matching_maps.sort(key=lambda x: (x[1]['subject_id'], x[1]['session_id'], x[0]))
    
    if len(matching_maps) == 0:
        st.warning(f"No maps found for this selection")
        st.stop()
    
    st.info(f"Found {len(matching_maps)} maps from {len(selected_sources)} source(s)")
    
    # Plot controls
    st.markdown("---")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        n_slices = st.number_input("# Slices", 0, 20, 7, key="n_slices_contrast")
    with ctrl_col2:
        use_symmetric = st.checkbox("Symmetric colorbar", value=True, key="symmetric_contrast")
    with ctrl_col3:
        vmax_contrast = st.number_input("vmax", value=5.0, format="%.1f", key="vmax_contrast")
    
    vmin_contrast = -vmax_contrast if use_symmetric else 0.0
    
    # Compute cut coordinates
    cut_coords = None
    if masker and n_slices > 0:
        try:
            mask_img = masker.mask_img_
            affine = mask_img.affine
            z_bounds = (affine[2, 3], affine[2, 3] + affine[2, 2] * mask_img.shape[2])
            cut_coords = np.linspace(z_bounds[0], z_bounds[1], n_slices + 2)[1:-1]
        except:
            cut_coords = None
    
    st.markdown("---")
    st.markdown("### 🗺️ All Maps")
    
    # Show all maps stacked vertically
    for source_name, map_info in matching_maps:
        subj_id = map_info['subject_id']
        sess_id = map_info['session_id']
        
        # Create a container for this map
        with st.container():
            # Header with source and subject/session info
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {source_name} | Subject {subj_id} | Session {sess_id}")
            
            with col2:
                # Show compact metadata with color coding
                fd_metrics = load_fd_metrics(subj_id, sess_id)
                if fd_metrics:
                    mean_fd = fd_metrics['mean_fd']
                    percent_bad = fd_metrics['percent_bad']
                    
                    # Color code based on FD quality
                    if mean_fd < 0.2:
                        st.success(f"🏃 FD: {mean_fd:.3f} mm ({percent_bad:.0f}% bad) - Excellent")
                    elif mean_fd < 0.5:
                        st.info(f"🏃 FD: {mean_fd:.3f} mm ({percent_bad:.0f}% bad) - Good")
                    elif mean_fd < 0.9:
                        st.warning(f"🏃 FD: {mean_fd:.3f} mm ({percent_bad:.0f}% bad) - Moderate")
                    else:
                        st.error(f"🏃 FD: {mean_fd:.3f} mm ({percent_bad:.0f}% bad) - High motion")
                else:
                    st.caption("🏃 FD: N/A")
            
            # Load behavioral data
            behav_values = get_behavioral_values(behavioral_df, subj_id, sess_id)
            if behav_values:
                behav_str = " | ".join([f"{k}: {v:.1f}" for k, v in list(behav_values.items())[:3]])
                st.markdown(f"**🧠 Cognitive:** {behav_str}")
            
            # Load and plot image
            img = load_nifti_image(map_info['file_path'])
            
            if img is None:
                st.error(f"Failed to load: {map_info['file_path']}")
            else:
                # Plot
                fig = plt.figure(figsize=(16, 4))
                try:
                    plot_stat_map(
                        img,
                        bg_img=None,
                        display_mode="z",
                        cut_coords=cut_coords,
                        cmap="cold_hot",
                        vmin=vmin_contrast,
                        vmax=vmax_contrast,
                        title=f"{source_name} | Sub {subj_id} Ses {sess_id}",
                        figure=fig,
                        symmetric_cbar=use_symmetric,
                        threshold=0.001,
                    )
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"Plotting error: {e}")
                    plt.close(fig)
                
                # Stats
                data = img.get_fdata().flatten()
                data = data[~np.isnan(data) & (data != 0)]
                if len(data) > 0:
                    st.caption(f"📈 Mean: {data.mean():.2f} | Std: {data.std():.2f} | Min: {data.min():.2f} | Max: {data.max():.2f}")
            
            st.markdown("---")

# ======================= FOOTER =======================

st.markdown("---")
st.caption("🔬 Activation Map Comparison Tool | fMRI Analysis Pipeline")
