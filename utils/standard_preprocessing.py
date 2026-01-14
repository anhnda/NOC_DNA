import pandas as pd
import numpy as np
import re

def preprocess_provedit_comprehensive(file_path, threshold=50, stutter_ratio=0.15):
    """
    Production-level preprocessing for PROVEDIt Evidence CSVs.
    - Handles up to 100 Allele/Size/Height triplets.
    - Flattens MultiIndex to prevent CSV header line breaks.
    - Removes technical artifacts (Noise, OL, Stutter).
    - Returns a clean, flat DataFrame with 'target_noc' as the final column.
    """
    
    # 1. LOAD DATA 
    # Use low_memory=False to avoid DtypeWarnings with wide CSVs
    print(f"Reading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 2. MELT WIDE FORMAT TO LONG
    # Standard PROVEDIt columns + any column starting with Allele/Size/Height
    id_vars = ['Sample File', 'Marker', 'Dye']
    triplet_cols = [col for col in df.columns if any(x in col for x in ['Allele', 'Size', 'Height'])]
    
    long_df = pd.melt(df, id_vars=id_vars, value_vars=triplet_cols, 
                      var_name='temp', value_name='Value')
    
    # Extract Attribute Type (Allele/Size/Height) and Peak Number
    # Regex handles potential spaces or lack thereof in "Allele 1" vs "Allele1"
    long_df[['Attribute', 'Peak_Num']] = long_df['temp'].str.extract(r'([a-zA-Z]+)\s*(\d+)')
    
    # Pivot to create a clean list of peaks per marker
    df_peaks = long_df.pivot_table(index=id_vars + ['Peak_Num'], 
                                   columns='Attribute', 
                                   values='Value', 
                                   aggfunc='first').reset_index()
    
    # 3. TYPE CASTING & CLEANING
    df_peaks['Height'] = pd.to_numeric(df_peaks['Height'], errors='coerce')
    df_peaks['Size'] = pd.to_numeric(df_peaks['Size'], errors='coerce')
    df_peaks['Allele'] = df_peaks['Allele'].astype(str).str.strip()
    
    # Drop empty rows and signals below the Analytical Threshold
    df_peaks = df_peaks.dropna(subset=['Allele', 'Height'])
    df_peaks = df_peaks[(df_peaks['Height'] >= threshold) & 
                        (df_peaks['Allele'] != 'nan') & 
                        (df_peaks['Allele'] != 'OL')].copy()

    # 4. HEURISTIC STUTTER FILTERING
    # Identifies small peaks (N-1) relative to the major peak in a locus
    df_peaks = df_peaks.sort_values(['Sample File', 'Marker', 'Height'], ascending=[True, True, False])
    
    def apply_stutter_filter(group):
        if len(group) < 2: return group
        major_size = group.iloc[0]['Size']
        major_height = group.iloc[0]['Height']
        # Flag peaks ~4bp smaller and < 15% of the height of the tallest peak
        is_stutter = (group['Size'] < major_size - 3.2) & \
                     (group['Size'] > major_size - 4.8) & \
                     (group['Height'] < major_height * stutter_ratio)
        return group[~is_stutter]

    df_peaks = df_peaks.groupby(['Sample File', 'Marker'], group_keys=False).apply(apply_stutter_filter)

    # 5. PIVOT TO GENOTYPE MATRIX (SPARSE FORMAT)
    # This creates the MultiIndex that causes the "New Line" problem
    final_df = df_peaks.pivot_table(index='Sample File', 
                                   columns=['Marker', 'Allele'], 
                                   values='Height', 
                                   fill_value=0)

    # 6. CRITICAL: FLATTEN AND CLEAN HEADERS
    # Convert (Marker, Allele) tuples into "Marker_Allele" strings
    final_df.columns = [f"{marker}_{allele}" for marker, allele in final_df.columns]
    
    # Remove the axis name to prevent the ghost header row in CSV
    final_df.columns.name = None
    
    # Reset index so 'Sample File' becomes a column, allowing NOC extraction
    final_df = final_df.reset_index()

    # 7. INJECT TARGET_NOC (GROUND TRUTH)
    def extract_noc(filename):
        """
        Precisely extracts the Number of Contributors from PROVEDIt filenames.
        Target: The segment AFTER the study ID (e.g., after RD14-0003)
        """
        # This regex looks for the pattern: STUDY_ID-CONTRIBUTORS-RATIO
        # Specifically: some chars, a hyphen, then the mix IDs, then another hyphen
        # Example match: -31_32-
        match = re.search(r'RD\d+-\d+-([\d_]+)-', filename)
        
        if match:
            contributor_part = match.group(1)
            # Split by underscore to count individuals
            contributors = contributor_part.split('_')
            return len(contributors)
        
        # Fallback: if the specific RD pattern isn't found, try a broader search
        # but skip the first two hyphenated groups
        parts = filename.split('-')
        if len(parts) > 2:
            # parts[0] is A02_RD14, parts[1] is 0003, parts[2] is 31_32
            contributors = parts[2].split('_')
            return len(contributors)
            
        return 1 # Final default
    final_df['target_noc'] = final_df['Sample File'].apply(extract_noc)

    return final_df

import pandas as pd
import numpy as np
import re

def preprocess_for_cnn(file_path, threshold=50, max_alleles=35):
    """
    Preprocessing optimized for CNN models.
    Converts DNA profiles into a 2D matrix (Image-like) of shape (Markers, Bins).
    """
    
    # 1. LOAD AND INITIAL CLEANING (Same as XGBoost logic)
    print(f"Processing for CNN: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    id_vars = ['Sample File', 'Marker', 'Dye']
    triplet_cols = [col for col in df.columns if any(x in col for x in ['Allele', 'Size', 'Height'])]
    
    long_df = pd.melt(df, id_vars=id_vars, value_vars=triplet_cols, var_name='temp', value_name='Value')
    long_df[['Attribute', 'Peak_Num']] = long_df['temp'].str.extract(r'([a-zA-Z]+)\s*(\d+)')
    
    df_peaks = long_df.pivot_table(index=id_vars + ['Peak_Num'], columns='Attribute', values='Value', aggfunc='first').reset_index()
    
    df_peaks['Height'] = pd.to_numeric(df_peaks['Height'], errors='coerce')
    df_peaks['Allele'] = df_peaks['Allele'].astype(str).str.strip()
    
    df_peaks = df_peaks.dropna(subset=['Allele', 'Height'])
    df_peaks = df_peaks[(df_peaks['Height'] >= threshold) & (df_peaks['Allele'] != 'OL')].copy()

    # 2. STANDARDIZING THE BINS
    # For a CNN, every 'image' must be the same size. 
    # We map Allele IDs to fixed integer bins (e.g., bin 0 to bin 35).
    def map_to_bin(allele_str):
        try:
            # Handle microvariants (9.3 -> 9.3)
            val = float(allele_str)
            # Map common alleles to a range (shifted for alignment)
            # You can customize this range based on your kit (e.g., 5 to 40)
            return int(round(val * 2)) # Multiplying by 2 handles .5/microvariants better
        except:
            if 'X' in allele_str: return 1
            if 'Y' in allele_str: return 2
            return 0

    df_peaks['Bin'] = df_peaks['Allele'].apply(map_to_bin)
    
    # 3. PIVOT TO 2D GRID (Markers x Bins)
    # We use Log-Scaling for heights to normalize intensity for the CNN
    df_peaks['LogHeight'] = np.log1p(df_peaks['Height'])

    # Create a 3D Tensor structure: (Samples, Markers, Bins)
    # For simplicity, we create a pivot where each row is a Sample-Marker pair
    cnn_matrix = df_peaks.pivot_table(index=['Sample File', 'Marker'], 
                                    columns='Bin', 
                                    values='LogHeight', 
                                    fill_value=0)

    # 4. ENSURE FIXED DIMENSIONS
    # Ensure every marker has exactly 'max_alleles' bins
    all_bins = list(range(max_alleles))
    for b in all_bins:
        if b not in cnn_matrix.columns:
            cnn_matrix[b] = 0.0
    
    cnn_matrix = cnn_matrix[all_bins] # Reorder columns to be sequential
    
    # 5. TARGET EXTRACTION (NOC)
    def extract_noc(filename):
        match = re.search(r'RD\d+-\d+-([\d_]+)-', filename)
        if match:
            return len(match.group(1).split('_'))
        parts = filename.split('-')
        if len(parts) > 2:
            return len(parts[2].split('_'))
        return 1

    # Flatten the matrix for export, but keep it structured so the CNN can reshape it
    # Format: Sample_File, Marker, Bin_0, Bin_1 ... Bin_N, target_noc
    final_df = cnn_matrix.reset_index()
    final_df['target_noc'] = final_df['Sample File'].apply(extract_noc)

    return final_df