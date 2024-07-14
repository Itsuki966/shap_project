import shap
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# 分析データの読み込み
df00_05 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='00_05', index_col=0)
df_05_10 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='05_10', index_col=0)
df_10_15 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='10_15', index_col=0)

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

# 増減率が50以上の項目を外れ値として除外
for i in col_list:
  all_data = all_data[all_data[i] < 50]

# SHAP値の読み込み
shap_values_young = np.load("/Users/itsukikuwahara/Desktop/shap_project/shap_values_young.npy")
shap_values_all = np.load("/Users/itsukikuwahara/Desktop/shap_project/shap_values_all.npy")


# 棒グラフによる特徴量重要度の可視化
# SHAP値の各項目絶対値の合計
sum_young_values = shap_values_young[0]
sum_all_values = shap_values_all[0]
for n in range(1, len(shap_values_young)):
  sum_young_values += np.abs(shap_values_young[n])

for i in range(1, len(shap_values_all)):
  sum_all_values += np.abs(shap_values_all[i])
  
# SHAP値の平均値
ave_shap_values = np.round(sum_young_values / len(shap_values_young), decimals=4)
ave_shap_values_all = np.round(sum_all_values / len(shap_values_all), decimals=4)

# 棒グラフのラベルの作成
labels = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values
label_list = list(labels)
ave_shap_list = list(ave_shap_values)
ave_shap_list_all = list(ave_shap_values_all)
zip_list = list(zip(ave_shap_list, ave_shap_list_all, labels))
zip_list.sort(reverse=False)
young = [zip_list[v][0] for v in range(len(labels))]
all = [zip_list[v][1] for v in range(len(labels))]
item_name = [zip_list[v][2] for v in range(len(labels))]

# 棒グラフのプロット
fig = plt.figure(figsize=[10,5])
# left = np.arange(0, 3*len(ave_shap_values), 3)
left = np.arange(0, 3*5, 3)
height = 1.35
# label = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values
label = [
  "Child welfare expenses",
  "Population commuting/going to school\nin their municipalities",
  "Number of workers in the tertiary industry",
  "Number of clinics/100,000",
  "Number of clinics/habitalbe area",
]
# label = item_name[-5:]

barh1 = plt.barh(
    y = left,
    width = young[-5:],
    height = height,
    label = 'young',
    color = 'black'
)

barh2 = plt.barh(
    y = left - height,
    width = all[-5:],
    height = height,
    label = 'all',
    color = 'lightgray',
    
    hatch = "/"
)

plt.bar_label(barh1, fontsize=20)
plt.bar_label(barh2, fontsize=20)
plt.yticks(ticks=left - height/2, labels=label, fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc=0, fontsize=20)
plt.title('mean(|SHAP Value|)', fontsize=20)
plt.show()


# beeswarm plotを使った可視化

# 若年層人口が増加している市町村の抽出とデータの加工
# 若年層人口が増加している都市のみのデータフレームを作成
# shap_df = all_data.mask(all_data['若年層人口'] <= 0)
# # young_df = shap_df.drop(["year", "area", "code","総人口", "若年層人口"], axis=1)

# # 若年層人口が増加している都市のみのデータフレームをtorch.tensor型に変更
# young_input = torch.tensor(
#     shap_df.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
#     dtype=torch.float32
# )

# young_target = torch.tensor(
#     shap_df['若年層人口'].values.astype(np.float32),
#     dtype=torch.float32
# )

# # 全ての市町村のデータもtorch.tensor型に変更
# all_input = torch.tensor(
#     all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
#     dtype=torch.float32,
# )

# # 若年層人口が増加している市町村のExplanationオブジェクト
# Explanation_young = shap.Explanation(
#     values = shap_values_young,
#     # base_values = expected_value,
#     data = young_input,
#     # display_data = shap_df,
#     feature_names = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values,
# )

# print(f"shap_values_young:{len(shap_values_young)}, young_inut:{len(young_input)}")

# # 全ての市町村のExplanationオブジェクト
# Explanation_all = shap.Explanation(
#     values = shap_values_all,
#     # base_values = explainer.expected_value,
#     data = all_input,
#     # display_data = shap_df
#     feature_names = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values
# )

# # 結果のプロット(若年層)
# shap.plots.beeswarm(
#     Explanation_young,
#     max_display=5,
#     plot_size = (25,20),
#     show = False
# )

# ax = plt.gca()
# ax.tick_params(labelsize=40)
# ax.set_xlabel('SHAP values (impact on model output)', fontsize=44)

# # 結果のプロット（全ての市町村）
# shap.plots.beeswarm(
#     Explanation_all,
#     max_display=5,
#     plot_size = (25,20),
#     show = False
# )

# ax = plt.gca()
# ax.tick_params(labelsize=40)
# ax.set_xlabel('SHAP values (impact on model output)', fontsize=44)
