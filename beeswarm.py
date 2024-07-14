import pandas as pd
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
# import japanize_matplotlib

# 分析データの読み込み
df00_05 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='00_05', index_col=0)
df_05_10 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='05_10', index_col=0)
df_10_15 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='10_15', index_col=0)

# SHAP値の読み込み
shap_values_young = np.load("/Users/itsukikuwahara/Desktop/shap_project/shap_values_young.npy")
shap_values_all = np.load("/Users/itsukikuwahara/Desktop/shap_project/shap_values_all.npy")

df_alldata = pd.concat([df00_05, df_05_10, df_10_15])
all_data = df_alldata.reset_index(drop=True)

col_list = ['歳出決算額', '社会福祉費', '老人福祉費', '児童福祉費',
       '農林水産業費', '商工費', '都市計画費', '住宅費', '教育費', '小学校費', '中学校費', '高校費',
       '幼稚園費', '小学校数', '小学校教員数', '中学校数', '中学校教員数', '高校数', '第一次産業就業者',
       '第二次産業就業者', '第三次産業就業者', '介護老人福祉施設(65歳以上人口10万人当たり)', '一般病院数/10万人',
       '一般診療所数/10万人', '一般病院数/可住地面積', '一般診療所数/可住地面積', '自市区町村で従業・通学している人口',
       '流出人口（県内他市区町村で従業・通学している人口）', '流出人口（他県で従業・通学している人口）',
       '流入人口（県内他市区町村に常住している人口）', '流入人口（他県に常住している人口）', '耕地面積【ｈａ】', '総人口',
       '昼夜間人口比率(%)', '財政力指標']

label_list = [
    '歳出決算額', '社会福祉費', '老人福祉費', 'Child welfare expenses',
    '農林水産業費', '商工費', '都市計画費', '住宅費', '教育費', '小学校費', '中学校費', '高校費',
    '幼稚園費', '小学校数', '小学校教員数', '中学校数', '中学校教員数', '高校数', '第一次産業就業者',
    '第二次産業就業者', 'Number of workers in the tertiary industry', '介護老人福祉施設(65歳以上人口10万人当たり)', '一般病院数/10万人',
    'Number of clinics/100,000', '一般病院数/可住地面積', 'Number of clinics/habitalbe area', 'Population commuting/going to school\nin their municipalities',
    '流出人口（県内他市区町村で従業・通学している人口）', '流出人口（他県で従業・通学している人口）',
    '流入人口（県内他市区町村に常住している人口）', '流入人口（他県に常住している人口）', '耕地面積【ｈａ】', '総人口',
    '昼夜間人口比率(%)', '財政力指標'
]

# 増減率が50以上の項目を外れ値として除外
for i in col_list:
  all_data = all_data[all_data[i] < 50]


# beeswarm plotを使った可視化

# 若年層人口が増加している市町村の抽出とデータの加工
# 若年層人口が増加している都市のみのデータフレームを作成
shap_df = all_data.mask(all_data['若年層人口'] <= 0)
young_df = shap_df.dropna()

# 若年層人口が増加している都市のみのデータフレームをtorch.tensor型に変更
young_input = torch.tensor(
    young_df.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
    dtype=torch.float32
)

# 全ての市町村のデータもtorch.tensor型に変更
all_input = torch.tensor(
    all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
    dtype=torch.float32,
)

# 若年層人口が増加している市町村のExplanationオブジェクト
Explanation_young = shap.Explanation(
    values = shap_values_young,
    data = young_input,
    # feature_names = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values,
    feature_names = label_list
)

# 全ての市町村のExplanationオブジェクト
Explanation_all = shap.Explanation(
    values = shap_values_all,
    data = all_input,
    # feature_names = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values
    feature_names = label_list
)

# 結果のプロット(若年層)
shap.plots.beeswarm(
    Explanation_young,
    max_display=6,
    plot_size = (10,6),
    show = False,
    color = plt.get_cmap('Greys')
)

ax = plt.gca()
ax.tick_params(labelsize=20)
ax.set_xlabel('SHAP values (impact on model output)', fontsize=20)
plt.show()

# 結果のプロット（全ての市町村）
shap.plots.beeswarm(
    Explanation_all,
    max_display=6,
    plot_size = (10,6),
    show = False,
    color = plt.get_cmap('Greys')
)

ax = plt.gca()
ax.tick_params(labelsize=20)
ax.set_xlabel('SHAP values (impact on model output)', fontsize=20)
plt.show()