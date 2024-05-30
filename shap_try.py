import shap
# import japanize_matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
# import scipy.stats
from sklearn import preprocessing

df00_05 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='00_05', index_col=0)
df_05_10 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='05_10', index_col=0)
df_10_15 = pd.read_excel('/Users/itsukikuwahara/Desktop/shap_project/processed_data_main.xlsx', sheet_name='10_15', index_col=0)

df_alldata = pd.concat([df00_05, df_05_10, df_10_15])
all_data = df_alldata.reset_index(drop=True)
print(len(all_data.columns))

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

# データの正規化
# mm = preprocessing.MinMaxScaler()


device = 'cpu'

# データをPyTorchでの学習に利用できる形式に変換
# "tip"の列を目的にする（tensor型に変換する際に正規化を行う）
# target = torch.tensor(mm.fit_transform(all_data['若年層人口'].values.reshape(-1, 1)), dtype=torch.float32, device=device)
target = torch.tensor(all_data["若年層人口"].values.reshape(-1, 1), dtype=torch.float32, device=device)
# "tip"以外の列を入力にする（tensor型に変換する際に正規化を行う）
input = torch.tensor(
    all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
    dtype=torch.float32,
    device=device,
)

# データセットの作成
all_data_dataset = torch.utils.data.TensorDataset(input, target)

# 学習データ、検証データ、テストデータに 6:2:2 の割合で分割
train_size = int(0.8 * len(all_data_dataset))
val_size = int(0.2 * len(all_data_dataset))
test_size = len(all_data_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    all_data_dataset, [train_size, val_size, test_size]
)

# バッチサイズ：25として学習用データローダを作成
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
# 検証用ローダ作成
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25)
# テスト用ローダを作成
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25)


# 3層順方向ニューラルネットワークモデル
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        o = self.l2(h)
        return o


# NNのオブジェクトを作成
model = SimpleNN(34, 30, 1).to(device)
# オプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# 損失の可視化のために各エポックの損失を保持しておくリスト
train_loss_list = []
test_loss_list = []
# データセット全体に対して10000回学習
for epoch in range(10000):
    epoch_loss = []
    # バッチごとに学習する
    for x, y_hat in train_loader:
        y = model(x)
        train_loss = torch.nn.functional.mse_loss(y, y_hat)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        epoch_loss.append(train_loss)
    # print(epoch_loss)
    train_loss_list.append(torch.tensor(epoch_loss).mean())

    with torch.inference_mode():  # 推論モード（学習しない）
        y = model(input)
        test_loss = torch.nn.functional.mse_loss(y, target)
        test_loss_list.append(test_loss.mean())

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, train_loss: {train_loss_list[-1]}, test_loss: {test_loss_list[-1]}")
        

# モデルの保存
# torch.save(model.state_dict(), 'simplenn_lr-00001_epoch10000.pt')

# 若年層人口が増加している都市のみのデータフレームを作成
shap_df = all_data.mask(all_data['若年層人口'] <= 0)
shap_df.dropna(inplace=True)

# 若年層人口が増加している都市のみのデータフレームをtorch.tensor型に変更
young_input = torch.tensor(
    shap_df.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
    dtype=torch.float32
)

young_target = torch.tensor(
    shap_df['若年層人口'].values.astype(np.float32),
    dtype=torch.float32
)

all_input = torch.tensor(
    all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).values.astype(np.float32),
    dtype=torch.float32,
)

# Torch Script形式で読み込み
# model = torch.load('/Users/itsukikuwahara/Desktop/simple_nn_project/simplenn_lr-00001_epoch10000.pt', map_location="cpu")

# SHAPのexplainerの作成
explainer = shap.DeepExplainer(model, input)
# すべての市町村のSHAP値の算出
shap_values_all = explainer.shap_values(input, check_additivity=False)
#若年層人口が増加している市町村に対してのSHAP値の算出
shap_values_young = explainer.shap_values(young_input, check_additivity=False)

# SHAP値の保存
np.save('shap_values_all', shap_values_all)
np.save("shap_values_young", shap_values_young)
np.save('shap_values_expected', explainer.expected_value)


# SHAP値の各項目絶対値の合計
sum_young_values = shap_values_young[0]
sum_all_values = shap_values_all[0]
for n in range(1, len(shap_values_young)):
  sum_young_values += np.abs(shap_values_young[n])

for i in range(1, len(shap_values_all)):
  sum_all_values += np.abs(shap_values_all[i])

# 平均を算出
ave_shap_values = np.round(sum_young_values / len(shap_values_young), decimals=4)
ave_shap_values_all = np.round(sum_all_values / len(shap_values_all), decimals=4)

# 列名(項目)とave_shap_valuesとave_shap_values_allを結びつける
labels = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values
label_list = list(labels)
ave_shap_list = list(ave_shap_values)
ave_shap_list_all = list(ave_shap_values_all)
zip_list = list(zip(ave_shap_list, ave_shap_list_all, label_list))

# ソートする
zip_list.sort(reverse=False)

# print(zip_list)

# # グラフにプロット
# young = [v[0] for v in zip_list]
# all = [v[1] for v in zip_list]
# item_name = [v[2] for v in zip_list]

# fig = plt.figure(figsize=[25,60])
# left = np.arange(0, 3*len(ave_shap_values), 3)
# height = 1.35
# label = item_name

# barh1 = plt.barh(
#     y = left,
#     width = young,
#     height = height,
#     label = 'young',
#     color = 'blue'
# )

# barh2 = plt.barh(
#     y = left - height,
#     width = all,
#     height = height,
#     label = 'all',
#     color = 'red',

# )

# plt.bar_label(barh1, fontsize=30)
# plt.bar_label(barh2, fontsize=30)
# plt.yticks(ticks=left - height/2, labels=label, fontsize=35)
# plt.xticks(fontsize=50)
# plt.legend(loc=0, fontsize=40)
# plt.title('mean(|SHAP Value|)', fontsize=50)
# fig.show()

Explanation_young = shap.Explanation(
    values = shap_values_young,
    # base_values = expected_value,
    data = young_input,
    # display_data = shap_df,
    feature_names = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values,
)

Explanation_all = shap.Explanation(
    values = shap_values_all,
    # base_values = explainer.expected_value,
    data = all_input,
    # display_data = shap_df
    feature_names = all_data.drop(["year", "area", "code","総人口", "若年層人口"], axis=1).columns.values
)

shap.plots.beeswarm(
    Explanation_all,
    max_display=5,
    plot_size = (25,20),
    show = False
)

ax = plt.gca()
ax.tick_params(labelsize=40)
ax.set_xlabel('SHAP values (impact on model output)', fontsize=44)
# ax.set_title(label='SHAP values (impact on model output)', fontsize=40)
# ax.set_xlim(-0.25,0.25)