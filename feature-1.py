import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import datetime
import warnings
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import sklearn
import gc

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

def kalman_smooth(x):

    series = [x['sales_0'], x['sales_1'], x['sales_2'], x['sales_3'], x['sales_4'],
                      x['sales_5'], x['sales_6'], x['sales_7'], x['sales_8'], x['sales_9'], x['sales_10'], x['sales_11'],
                      x['sales_12'], x['sales_13'], x['sales_14']]
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=1, initial_state_mean=series[0])
    state_means, state_covariance = kf.smooth(series)
    return state_means.ravel().tolist()

def train_model():
    df = pd.read_csv('fusai_data/goodsale_modified1.csv')
    df['data_date'] = pd.to_datetime(df['data_date'], format='%Y-%m-%d')
    sub = pd.read_csv('fusai_data/submit_example_2.csv')
    info = pd.read_csv('fusai_data/goodsinfo.csv')
    relation = pd.read_csv('fusai_data/goods_sku_relation.csv')
    relation = pd.merge(relation, info, on='goods_id')

    df['goods_price'] = df['goods_price'].map(lambda x: x.replace(',', '') if type(x) == np.str else x)
    df['goods_price'] = pd.to_numeric(df['goods_price'])
    df['orginal_shop_price'] = df['orginal_shop_price'].map(lambda x: x.replace(',', '') if type(x) == np.str else x)
    df['orginal_shop_price'] = pd.to_numeric(df['orginal_shop_price'])


    daily = pd.read_csv('fusai_data/daily_modified1.csv')
    daily['data_date'] = pd.to_datetime(daily['data_date'], format='%Y-%m-%d')
    droped = daily.drop_duplicates(subset='goods_id')
    droped['open_date'] = droped.apply(lambda x: x['data_date'] - datetime.timedelta(x['onsale_days']), axis=1)

    grouped = df.groupby(['sku_id', 'own_week'])['goods_num'].sum().reset_index()
    pivot = grouped.pivot(index='sku_id', columns='own_week', values='goods_num')
    new_columns = {}
    for i in list(pivot.columns):
        new_columns[i] = 'sales_' + str(i)
    pivot.rename(columns=new_columns, inplace=True)
    pivot.fillna(0, inplace=True)


    grouped_daily = daily.groupby(['goods_id', 'own_week'])['goods_click'].sum().reset_index()
    pivot_daily = grouped_daily.pivot(index='goods_id', columns='own_week', values='goods_click')
    new_columns = {}
    for i in list(pivot_daily.columns):
        new_columns[i] = 'goods_click_' + str(i)
    pivot_daily.rename(columns=new_columns, inplace=True)
    pivot_daily.fillna(0, inplace=True)

    grouped_daily_cart = daily.groupby(['goods_id', 'own_week'])['cart_click'].sum().reset_index()
    pivot_daily_cart = grouped_daily_cart.pivot(index='goods_id', columns='own_week', values='cart_click')
    new_columns = {}
    for i in list(pivot_daily_cart.columns):
        new_columns[i] = 'cart_click_' + str(i)
    pivot_daily_cart.rename(columns=new_columns, inplace=True)
    pivot_daily_cart.fillna(0, inplace=True)

    grouped_daily_fav = daily.groupby(['goods_id', 'own_week'])['favorites_click'].sum().reset_index()
    pivot_daily_fav = grouped_daily_fav.pivot(index='goods_id', columns='own_week', values='favorites_click')
    new_columns = {}
    for i in list(pivot_daily_fav.columns):
        new_columns[i] = 'favorites_click_' + str(i)
    pivot_daily_fav.rename(columns=new_columns, inplace=True)
    pivot_daily_fav.fillna(0, inplace=True)

    grouped_daily_uv = daily.groupby(['goods_id', 'own_week'])['sales_uv'].sum().reset_index()
    pivot_daily_uv = grouped_daily_uv.pivot(index='goods_id', columns='own_week', values='sales_uv')
    new_columns = {}
    for i in list(pivot_daily_uv.columns):
        new_columns[i] = 'sales_uv_' + str(i)
    pivot_daily_uv.rename(columns=new_columns, inplace=True)
    pivot_daily_uv.fillna(0, inplace=True)

    sub = pd.merge(sub, pivot, on='sku_id', how='left')
    sub = pd.merge(sub, relation, on='sku_id', how='left')
    sub = pd.merge(sub, pivot_daily, on='goods_id', how='left')
    sub = pd.merge(sub, pivot_daily_cart, on='goods_id', how='left')
    sub = pd.merge(sub, pivot_daily_fav, on='goods_id', how='left')
    sub = pd.merge(sub, pivot_daily_uv, on='goods_id', how='left')

    sub = pd.merge(sub, droped[['goods_id', 'open_date']], on='goods_id', how='left')
    sub['onsale_train'] = sub['open_date'].map(lambda x: (datetime.datetime(2018, 3, 16) - x).days)
    sub['onsale_test'] = sub['open_date'].map(lambda x: (datetime.datetime(2018, 5, 7) - x).days)
    sub['concat'] = sub.apply(lambda x: str(x['cat_level1_id']) +
                                        '_' + str(x['cat_level2_id']) + '_' + str(x['cat_level3_id'])
                                        + '_' + str(x['cat_level4_id']) + '_' + str(x['cat_level5_id']), axis=1)


    raw_price = df.groupby('sku_id')['orginal_shop_price'].mean().reset_index()
    real_price = df.groupby('sku_id')['goods_price'].mean().reset_index()


    sub = pd.merge(sub, raw_price, on='sku_id', how='left')
    sub = pd.merge(sub, real_price, on='sku_id', how='left')

    sub['discount'] = sub['orginal_shop_price'] - sub['goods_price']

    print('------------load_data-----------------')

    sub['smooth'] = sub.apply(lambda x: kalman_smooth(x), axis=1)
    for i in range(15):
        sub['sales_smo_'+str(i)] = sub.apply(lambda x: x['smooth'][i], axis=1)
    print('------------kalman smooth-----------------')

    sub['sales_8'] = sub['sales_8'] * 1.1
    sub['sales_9'] = sub['sales_9'] * 1.2
    sub['sales_10'] = sub['sales_10'] * 1
    sub['sales_11'] = sub['sales_11'] * 0.6
    sub['sales_12'] = sub['sales_12'] * 0.7
    sub['sales_13'] = sub['sales_13'] * 0.8
    sub['sales_14'] = sub['sales_14'] * 0.9


    trian_features = ['sku_id', 'goods_id', 'brand_id', 'goods_season', 'cat_level1_id', 'concat',  'orginal_shop_price', 'goods_price', 'discount']
    test_features = ['sku_id', 'goods_id', 'brand_id', 'cat_level1_id', 'concat', 'goods_season',  'orginal_shop_price', 'goods_price', 'discount']
    trian_features.append('onsale_train')
    test_features.append('onsale_test')

    for i in range(0, 8):
        test_features.append('sales_' + str(i))
        test_features.append('sales_smo_'+str(i))
        test_features.append('goods_click_' + str(i))
        test_features.append('cart_click_' + str(i))
        test_features.append('favorites_click_' + str(i))
        test_features.append('sales_uv_'+str(i))
    for i in range(7, 15):
        trian_features.append('sales_' + str(i))
        trian_features.append('sales_smo_'+str(i))
        trian_features.append('goods_click_' + str(i))
        trian_features.append('cart_click_' + str(i))
        trian_features.append('favorites_click_' + str(i))
        trian_features.append('sales_uv_'+str(i))

    X_train = sub[trian_features]
    y_train = sub[['sales_0', 'sku_id', 'goods_id']]
    X_test = sub[test_features]

    X_train['onsale'] = X_train['onsale_train']
    X_test['onsale'] = X_test['onsale_test']



    X_train['sales_win_0'] = 0
    X_test['sales_win_0'] = 0
    X_train['click_win_0'] = 0
    X_test['click_win_0'] = 0
    X_train['cart_win_0'] = 0
    X_test['cart_win_0'] = 0
    X_train['favorites_win_0'] = 0
    X_test['favorites_win_0'] = 0
    X_train['uv_win_0'] = 0
    X_test['uv_win_0'] = 0
    all_features = ['guize', 'mean_sale', 'median_sale', 'goods_price', 'discount', 'onsale']

    guize_type = 'sales_smo_'
    X_train['guize'] = (13*X_train[guize_type+'7'] + 7*X_train[guize_type+'8'] + 6*X_train[guize_type+'9'] + 5*X_train[guize_type+'10']+
                        4*X_train[guize_type+'11']+ 3*X_train[guize_type+'12']+2*X_train[guize_type+'12']+X_train[guize_type+'14'])/41
    X_test['guize'] = (13 * X_test[guize_type+'0'] + 7 * X_test[guize_type+'1'] + 6 * X_test[guize_type+'2'] + 5 * X_test[
        guize_type+'3'] +4 * X_test[guize_type+'4'] + 3 * X_test[guize_type+'5'] + 2 * X_test[guize_type+'6'] + X_test[guize_type+'7'])/41

    sales_type = 'sales_'
    X_train['mean_sale'] = X_train.apply(
        lambda x: np.mean([x[sales_type+'7'], x[sales_type+'8'], x[sales_type+'9'], x[sales_type+'10'], x[sales_type+'11'],
                      x[sales_type+'12'], x[sales_type+'13'], x[sales_type+'14']]), axis=1)
    X_test['mean_sale'] = X_test.apply(
        lambda x: np.mean([x[sales_type+'0'], x[sales_type+'1'], x[sales_type+'2'], x[sales_type+'3'], x[sales_type+'4'],
                      x[sales_type+'5'], x[sales_type+'6'], x[sales_type+'7']]), axis=1)
    X_train['median_sale'] = X_train.apply(
        lambda x: np.median([x[sales_type+'7'], x[sales_type+'8'], x[sales_type+'9'], x[sales_type+'10'], x[sales_type+'11'],
                      x[sales_type+'12'], x[sales_type+'13'], x[sales_type+'14']]), axis=1)
    X_test['median_sale'] = X_test.apply(
        lambda x: np.median([x[sales_type+'0'], x[sales_type+'1'], x[sales_type+'2'], x[sales_type+'3'], x[sales_type+'4'],
                      x[sales_type+'5'], x[sales_type+'6'], x[sales_type+'7']]), axis=1)


    for i in range(1, 9):
        X_train['sales_win_' + str(i)] = X_train['sales_' + str(i + 6)] + X_train['sales_win_' + str(i - 1)]
        X_train['click_win_' + str(i)] = X_train['goods_click_' + str(i + 6)]
        X_train['cart_win_' + str(i)] = X_train['cart_click_' + str(i + 6)] + X_train['cart_win_' + str(i - 1)]
        X_train['favorites_win_' + str(i)] = X_train['favorites_click_' + str(i + 6)] + X_train['favorites_win_' + str(i - 1)]
        X_train['uv_win_' + str(i)] = X_train['sales_uv_' + str(i + 6)] + X_train['uv_win_' + str(i - 1)]

        X_train['sales/click_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['click_win_' + str(i)]
        X_train = X_train.join(X_train.groupby('goods_id')['sales_win_' + str(i)].sum().rename('goods_sales_win_' + str(i)), on='goods_id')
        X_train = X_train.join(X_train.groupby('goods_id')['click_win_' + str(i)].sum().rename('goods_click_win_' + str(i)), on='goods_id')

        X_train = X_train.join(X_train.groupby('cat_level1_id')['sales_win_' + str(i)].sum().rename('cat1_sales_win_' + str(i)), on='cat_level1_id')
        X_train = X_train.join(X_train.groupby('concat')['sales_win_' + str(i)].sum().rename('concat_sales_win_' + str(i)), on='concat')
        X_train = X_train.join(X_train.groupby('concat')['sales_win_' + str(i)].mean().rename('concat_sales_win_' + str(i)+'_mean'), on='concat')

        X_train = X_train.join(X_train.groupby('brand_id')['sales_win_' + str(i)].sum().rename('brand_sales_win_' + str(i)), on='brand_id')

        X_train['goods/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['goods_sales_win_' + str(i)]
        X_train['brand/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['brand_sales_win_' + str(i)]

        X_train['goods_click/sku_win' + str(i)] = X_train['click_win_' + str(i)] / X_train['goods_click_win_' + str(i)]

        X_train['cat1/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['cat1_sales_win_' + str(i)]
        X_train['concat/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['concat_sales_win_' + str(i)]
        X_train['concat-sku_win' + str(i)] = X_train['sales_win_' + str(i)] - X_train['concat_sales_win_' + str(i)+'_mean']


        X_train = X_train.join(X_train.groupby('concat')['sales_win_' + str(i)].rank().rename('concat/sku_rank_win' + str(i)))

        all_features.append('sales_win_' + str(i))


        all_features.append('goods/sku_win' + str(i))

        all_features.append('concat/sku_win'+str(i))



    for i in range(1, 9):
        X_test['sales_win_' + str(i)] = X_test['sales_' + str(i - 1)] + X_test['sales_win_' + str(i - 1)]

        X_test['click_win_' + str(i)] = X_test['goods_click_' + str(i - 1)]
        X_test['cart_win_' + str(i)] = X_test['cart_click_' + str(i - 1)] + X_test['cart_win_' + str(i - 1)]
        X_test['favorites_win_' + str(i)] = X_test['favorites_click_' + str(i - 1)] + X_test['favorites_win_' + str(i - 1)]
        X_test['uv_win_' + str(i)] = X_test['sales_uv_' + str(i - 1)] + X_test['uv_win_' + str(i - 1)]

        X_test['sales/click_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['click_win_' + str(i)]
        X_test = X_test.join(X_test.groupby('goods_id')['sales_win_' + str(i)].sum().rename('goods_sales_win_' + str(i)), on='goods_id')
        X_test = X_test.join(X_test.groupby('goods_id')['click_win_' + str(i)].sum().rename('goods_click_win_' + str(i)), on='goods_id')

        X_test = X_test.join(X_test.groupby('cat_level1_id')['sales_win_' + str(i)].sum().rename('cat1_sales_win_' + str(i)), on='cat_level1_id')
        X_test = X_test.join(X_test.groupby('concat')['sales_win_' + str(i)].sum().rename('concat_sales_win_' + str(i)), on='concat')
        X_test = X_test.join(X_test.groupby('brand_id')['sales_win_' + str(i)].sum().rename('brand_sales_win_' + str(i)), on='brand_id')
        X_test = X_test.join(X_test.groupby('concat')['sales_win_' + str(i)].mean().rename('concat_sales_win_' + str(i)+'_mean'), on='concat')

        X_test['goods/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['goods_sales_win_' + str(i)]
        X_test['goods_click/sku_win' + str(i)] = X_test['click_win_' + str(i)] / X_test['goods_click_win_' + str(i)]
        X_test['brand/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['brand_sales_win_' + str(i)]

        X_test['cat1/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['cat1_sales_win_' + str(i)]
        X_test['concat/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['concat_sales_win_' + str(i)]
        X_test['concat-sku_win' + str(i)] = X_test['sales_win_' + str(i)] - X_test['concat_sales_win_' + str(i)+'_mean']

        X_test = X_test.join(X_test.groupby('concat')['sales_win_' + str(i)].rank().rename('concat/sku_rank_win' + str(i)))


    print('------------generate features-----------------')

    clf = xgb.XGBRegressor(random_state=666, n_estimators=70, silent=False, n_jobs=4)
    clf.fit(X_train[all_features], y_train['sales_0'])
    print(pd.Series(clf.feature_importances_, all_features))
    y_pred = clf.predict(X_test[all_features])

    season_x = X_test['goods_season'].map(lambda x: 0.5 if x == 4 else 1)
    y_pred = y_pred * season_x
    season_x = X_test['goods_season'].map(lambda x: 0.9 if x == 2 else 1)
    y_pred = y_pred * season_x
    season_x = X_test['goods_season'].map(lambda x: 0.9 if x == 1 else 1)
    y_pred = y_pred * season_x
    season_x = X_test['goods_season'].map(lambda x: 0.9 if x == 3 else 1)
    y_pred = y_pred * season_x

    sub['week3'] = y_pred * 1.6

    sub['week1'] = sub['week3'].map(lambda x: (x / 1.6) * 1)
    sub['week2'] = sub['week3'].map(lambda x: (x / 1.6) * 1.3)
    sub['week3'] = sub['week3'].map(lambda x: (x / 1.6) * 1.7)
    sub['week4'] = sub['week3'].map(lambda x: (x / 1.6) * 2.1)
    sub['week5'] = sub['week3'].map(lambda x: (x / 1.6) * 0.7)

    print('------------predict-----------------')

    sub[['sku_id', 'week1', 'week2', 'week3', 'week4', 'week5']].to_csv('fusai_data/1112.csv', index=False)
    print('------------ok!!!!!!!!!!!-----------------')

def pre_process():
    sale = pd.read_csv('fusai_data/goodsale.csv')
    sub = pd.read_csv('fusai_data/submit_example_2.csv')
    daily = pd.read_csv('fusai_data/goodsdaily.csv')
    relation = pd.read_csv('fusai_data/goods_sku_relation.csv')

    sale['data_date'] = pd.to_datetime(sale['data_date'], format='%Y%m%d')
    sale['own_week'] = sale['data_date'].map(lambda x: (datetime.datetime(2018, 3, 16)-x).days//7)
    sale.to_csv('fusai_data/goodsale_modified1.csv', index=False)

    print('-----------------生成sale数据ok----------------')
    sub = pd.merge(sub, relation, on='sku_id', how='left')
    part = daily[daily['goods_id'].isin(sub['goods_id'].unique())]
    part['data_date'] = pd.to_datetime(part['data_date'], format='%Y%m%d')
    part['own_week'] = part['data_date'].map(lambda x: (datetime.datetime(2018, 3, 16) - x).days//7)
    part.to_csv('fusai_data/daily_modified1.csv', index=False)
    print('-----------------生成daily数据ok----------------')


if __name__ == '__main__':
    pre_process()
    train_model()
