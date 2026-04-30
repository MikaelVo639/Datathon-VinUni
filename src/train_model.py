import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("=== PIPELINE DỰ BÁO DOANH THU E-COMMERCE (DATATHON 2026) ===")

    # 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
    print("[1/5] Đang xử lý dữ liệu và lịch Khuyến mãi...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')

    sales = pd.read_csv(os.path.join(data_dir, 'sales.csv'), parse_dates=['Date'])
    promotions = pd.read_csv(os.path.join(data_dir, 'promotions.csv'), parse_dates=['start_date', 'end_date'])

    # Tạo lịch bao phủ toàn bộ thời gian Train + Test
    all_dates = pd.date_range(start=sales['Date'].min(), end='2024-07-01')
    df_master = pd.DataFrame({'Date': all_dates})
    df_master['has_promo'] = 0

    # Trải phẳng cờ Khuyến mãi
    for _, row in promotions.iterrows():
        mask = (df_master['Date'] >= row['start_date']) & (df_master['Date'] <= row['end_date'])
        df_master.loc[mask, 'has_promo'] = 1

    df_master = pd.merge(df_master, sales, on='Date', how='left')

    # DATA TRUNCATION: Loại bỏ kỷ nguyên cũ, chỉ tập trung vào hành vi từ 2019
    df_master = df_master[df_master['Date'] >= '2019-01-01'].reset_index(drop=True)

    # 2. FEATURE ENGINEERING (Domain Knowledge VN)
    print("[2/5] Trích xuất đặc trưng thời gian & Domain Knowledge...")
    df_master['day_of_week'] = df_master['Date'].dt.dayofweek
    df_master['day_of_month'] = df_master['Date'].dt.day
    df_master['day_of_year'] = df_master['Date'].dt.dayofyear
    df_master['month'] = df_master['Date'].dt.month
    df_master['year'] = df_master['Date'].dt.year
    df_master['is_weekend'] = df_master['day_of_week'].isin([5, 6]).astype(int)

    # Magic Features: Mua sắm Việt Nam
    df_master['is_double_day'] = (df_master['Date'].dt.day == df_master['Date'].dt.month).astype(int)
    df_master['is_payday'] = df_master['day_of_month'].isin([25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5]).astype(int)

    # 3. CHUẨN BỊ TRAIN / TEST
    train_df = df_master[df_master['Date'] <= '2022-12-31'].copy()
    test_df = df_master[df_master['Date'] >= '2023-01-01'].copy()

    features = [
        'day_of_week', 'day_of_month', 'day_of_year', 'month', 'year',
        'is_weekend', 'has_promo', 'is_double_day', 'is_payday'
    ]

    X_train = train_df[features]
    y_train_log = np.log1p(train_df['Revenue']) # Transform kéo RMSE xuống cực mạnh
    X_test = test_df[features]

    # 4. HUẤN LUYỆN ENSEMBLE MODEL (XGBoost + LightGBM)
    print("[3/5] Đang huấn luyện LightGBM...")
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
    }
    model_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=600)
    model_lgb.fit(X_train, y_train_log, categorical_feature=['day_of_week', 'month'])
    preds_lgb = np.expm1(model_lgb.predict(X_test))

    print("[4/5] Đang huấn luyện XGBoost...")
    xgb_params = {
        'objective': 'reg:squarederror', 'learning_rate': 0.03, 'max_depth': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
    }
    model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=600)
    model_xgb.fit(X_train, y_train_log)
    preds_xgb = np.expm1(model_xgb.predict(X_test))

    # 5. CHỐT KẾT QUẢ VÀ XUẤT FILE
    print("[5/5] Tính toán COGS và đóng gói file Submission...")
    # Trọng số: 60% LGBM (xử lý category tốt) + 40% XGBoost (kiểm soát nhiễu tốt)
    test_df['Revenue'] = (preds_lgb * 0.6) + (preds_xgb * 0.4)

    # Dự báo COGS bằng tỷ lệ biên lợi nhuận của năm gần nhất (2022)
    recent_cogs_ratio = train_df[train_df['year'] == 2022]['COGS'].sum() / train_df[train_df['year'] == 2022]['Revenue'].sum()
    test_df['COGS'] = test_df['Revenue'] * recent_cogs_ratio

    submission = test_df[['Date', 'Revenue', 'COGS']].copy()
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')

    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"=== HOÀN TẤT! File '{submission_file}' đã sẵn sàng ===")

if __name__ == "__main__":
    main()