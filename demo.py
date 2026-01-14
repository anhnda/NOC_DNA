from config import ONE_PERSON_PATH, MORE_PERSON_PATH
import pandas as pd

from utils.standard_preprocessing import preprocess_provedit_comprehensive
def preprocess():
    print("Preprocessing one-person sample...")
    one_person_df = preprocess_provedit_comprehensive(ONE_PERSON_PATH)
    print("One-person sample preprocessing complete.\n")
    
    print("Preprocessing more-person sample...")
    more_person_df = preprocess_provedit_comprehensive(MORE_PERSON_PATH)
    print("More-person sample preprocessing complete.\n")
    
    return one_person_df, more_person_df
if __name__ == "__main__":
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
    df_con.to_csv('./data/combined_preprocessed_data.csv', index=False)
    one_person_data.to_csv('./data/one_person_preprocessed_data.csv', index=False)
    more_person_data.to_csv('./data/more_person_preprocessed_data.csv', index=False)
    print("Data shapes:")
    print(f"One-person data shape: {one_person_data.shape}")
    print(f"More-person data shape: {more_person_data.shape}")
    print(f"Combined data shape: {df_con.shape}")