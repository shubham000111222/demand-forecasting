import os
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.features import BASE_DIR, ID_COL, SAMPLE_SUB_PATH, SUBMISSION_PATH

def main():
    start_time = time.time()
    
    print("1. Loading Models and Data...")
    
    with open(os.path.join(BASE_DIR, 'outputs', 'features.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    features = metadata['features']
    
    with open(os.path.join(BASE_DIR, 'outputs', 'ensemble_weights.pkl'), 'rb') as f:
        best_w = pickle.load(f)
        
    test = pd.read_pickle(os.path.join(BASE_DIR, 'outputs', 'test_processed.pkl'))
    X_test = test[features]
    
    final_lgb = lgb.Booster(model_file=os.path.join(BASE_DIR, 'outputs', 'lgb_model.txt'))
    
    final_cb = CatBoostRegressor()
    final_cb.load_model(os.path.join(BASE_DIR, 'outputs', 'cb_model.cbm'))
    
    print("2. Generating Blended Predictions...")
    lgb_test_preds = final_lgb.predict(X_test)
    cb_test_preds = final_cb.predict(X_test)
    
    blended_log_preds = best_w * lgb_test_preds + (1.0 - best_w) * cb_test_preds
    
    preds_expm1 = np.expm1(blended_log_preds)
    preds_clipped = np.maximum(preds_expm1, 0.0)
    
    test['OrderVolume'] = preds_clipped
    test.loc[test['IsOpen'] == 0, 'OrderVolume'] = 0.0
    
    print("3. Saving submission.csv...")
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    
    pred_dict = dict(zip(test[ID_COL], test['OrderVolume']))
    sub['OrderVolume'] = sub[ID_COL].map(pred_dict).fillna(0.0)
    
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print(f"Prediction completed in {(time.time() - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()
