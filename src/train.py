import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pickle

from src.features import (
    BASE_DIR, TARGET, CATEGORICAL_FEATURES, FEATURES_TO_DROP, VALIDATION_DAYS,
    preprocess_pipeline, feature_engineering_pipeline
)

def optimize_ensemble_weights(y_true, lgb_preds, cb_preds):
    def loss_func(weights):
        w1, w2 = weights
        blended = w1 * lgb_preds + w2 * cb_preds
        return np.sqrt(mean_squared_error(y_true, blended))
    
    initial_weights = [0.5, 0.5]
    bounds = ((0, 1), (0, 1))
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    result = minimize(loss_func, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    best_w = result.x[0]
    best_loss = result.fun
    
    print(f"Optimal Weights: {best_w:.3f} LightGBM, {1-best_w:.3f} CatBoost")
    print(f"Ensemble RMSLE: {best_loss:.5f}")
    return best_w, best_loss

def main():
    start_time = time.time()
    
    print("1. Preprocessing Data...")
    train, test = preprocess_pipeline()
    
    print("2. Generating Comprehensive Feature Set...")
    train, test = feature_engineering_pipeline(train, test)
    
    features = [c for c in train.columns if c not in FEATURES_TO_DROP]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in features]
    cat_indices = [features.index(c) for c in cat_feats]
    
    # Save processed test data for predict.py
    print("Saving processed test data...")
    test.to_pickle(os.path.join(BASE_DIR, 'outputs', 'test_processed.pkl'))
    with open(os.path.join(BASE_DIR, 'outputs', 'features.pkl'), 'wb') as f:
        pickle.dump({'features': features, 'cat_indices': cat_indices, 'cat_feats': cat_feats}, f)
    
    print("3. Validation Split...")
    max_date = train['Date'].max()
    val_cutoff = max_date - pd.Timedelta(days=VALIDATION_DAYS)
    
    train_split = train[train['Date'] <= val_cutoff].reset_index(drop=True)
    val_split = train[(train['Date'] > val_cutoff) & (train['Date'] <= max_date)].reset_index(drop=True)
    
    X_tr, y_tr = train_split[features], np.log1p(train_split[TARGET])
    X_va, y_va = val_split[features], np.log1p(val_split[TARGET])
    
    print("4A. Training LightGBM Model...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feats)
    val_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_feats, reference=trn_data)
    
    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    lgb_best_iter = lgb_model.best_iteration
    lgb_val_preds = lgb_model.predict(X_va, num_iteration=lgb_best_iter)
    
    print("4B. Training CatBoost Model...")
    cb_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': 0.05,
        'depth': 7,
        'random_seed': 42,
        'verbose': False
    }
    
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_indices,
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    cb_best_iter = cb_model.get_best_iteration()
    cb_val_preds = cb_model.predict(X_va)
    
    print("5. Optimizing Ensemble Weights...")
    best_w, _ = optimize_ensemble_weights(y_va.values, lgb_val_preds, cb_val_preds)
    
    print("6. Retraining Models on 100% Training Data...")
    X_full, y_full = train[features], np.log1p(train[TARGET])
    
    full_lgb_data = lgb.Dataset(X_full, label=y_full, categorical_feature=cat_feats)
    final_lgb = lgb.train(lgb_params, full_lgb_data, num_boost_round=int(lgb_best_iter * 1.05))
    
    full_cb_params = cb_params.copy()
    full_cb_params['iterations'] = int(cb_best_iter * 1.05)
    final_cb = CatBoostRegressor(**full_cb_params)
    final_cb.fit(X_full, y_full, cat_features=cat_indices, verbose=False)
    
    print("7. Saving Models...")
    final_lgb.save_model(os.path.join(BASE_DIR, 'outputs', 'lgb_model.txt'))
    final_cb.save_model(os.path.join(BASE_DIR, 'outputs', 'cb_model.cbm'))
    with open(os.path.join(BASE_DIR, 'outputs', 'ensemble_weights.pkl'), 'wb') as f:
        pickle.dump(best_w, f)
        
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
