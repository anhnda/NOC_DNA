from config import ONE_PERSON_PATH, MORE_PERSON_PATH, COMBINE_PREPROCESSED_CNN_PATH, COMBINE_PREPROCESSED_PATH
import pandas as pd

from utils.standard_preprocessing import preprocess_provedit_comprehensive, preprocess_for_cnn
def preprocess():
    print("Preprocessing one-person sample...")
    one_person_df = preprocess_provedit_comprehensive(ONE_PERSON_PATH)
    print("One-person sample preprocessing complete.\n")
    
    print("Preprocessing more-person sample...")
    more_person_df = preprocess_provedit_comprehensive(MORE_PERSON_PATH)
    print("More-person sample preprocessing complete.\n")
    
    return one_person_df, more_person_df
def export_combined_data_cls():
    one_person_data, more_person_data = preprocess()
    
    # 1. Align the dataframes
    # This ensures both DFs have the same columns in the same order
    print("Aligning columns for concatenation...")
    df_con = pd.concat([one_person_data, more_person_data], axis=0, sort=True)
    
    # 2. Fill NaNs with 0
    # Because a missing column in one file means that Allele had 0 Height (RFU)
    df_con = df_con.fillna(0)
    
    # 3. Move target_noc to the end (aesthetic and standard for ML)
    cols = [c for c in df_con.columns if c != 'target_noc'] + ['target_noc']
    df_con = df_con[cols]
    
    # Save results
    df_con.to_csv(COMBINE_PREPROCESSED_PATH, index=False)
    one_person_data.to_csv('./data/one_person_preprocessed_data.csv', index=False)
    more_person_data.to_csv('./data/more_person_preprocessed_data.csv', index=False)
    print("Data shapes:")
    print(f"One-person data shape: {one_person_data.shape}")
    print(f"More-person data shape: {more_person_data.shape}")
    print(f"Combined data shape: {df_con.shape}")
def export_combine_data_cnn():
    print("Preprocessing one-person sample for CNN...")
    one_person_cnn = preprocess_for_cnn(ONE_PERSON_PATH)
    
    print("Preprocessing more-person sample for CNN...")
    more_person_cnn = preprocess_for_cnn(MORE_PERSON_PATH)
    
    # 1. Combine the datasets
    # Since preprocess_for_cnn uses fixed bins, columns should already align
    print("Combining CNN datasets...")
    df_con_cnn = pd.concat([one_person_cnn, more_person_cnn], axis=0, ignore_index=True)
    
    # 2. Reshaping Logic
    # Currently, df_con_cnn has [Sample File, Marker, Bin_0...Bin_34, target_noc]
    # We need to ensure every Sample File has the same number of markers
    
    # Optional: Filter to only keep common markers (GlobalFiler standard is 24 markers)
    # df_con_cnn = df_con_cnn[df_con_cnn['Marker'].isin(list_of_24_markers)]

    # 3. Pivot to "Deep" format if you want a single row per sample
    # This creates a "flattened" image: [Sample File, M1_B0, M1_B1... M24_B34, target_noc]
    print("Flattening into 2D tensor format...")
    
    # We group by Sample and NOC, then unstack the Marker level
    # This turns the 'Marker' rows into extra columns
    cnn_final = df_con_cnn.set_index(['Sample File', 'target_noc', 'Marker']).unstack('Marker')
    
    # Collapse the multi-index header: (Bin_0, D3S1358) -> D3S1358_Bin_0
    cnn_final.columns = [f"{marker}_{bin_id}" for bin_id, marker in cnn_final.columns]
    cnn_final = cnn_final.reset_index()
    
    # 4. Final fill and target ordering
    cnn_final = cnn_final.fillna(0)
    cols = [c for c in cnn_final.columns if c != 'target_noc'] + ['target_noc']
    cnn_final = cnn_final[cols]
    
    # Save results
    cnn_final.to_csv(COMBINE_PREPROCESSED_CNN_PATH, index=False)
    print(f"CNN Combined data shape: {cnn_final.shape}")
    print("CNN Preprocessing complete.")

if __name__ == "__main__":
    # You can run both or choose one
    # export_combined_data_cls()
    export_combine_data_cnn()
