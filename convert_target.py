import copy

import pandas as pd
from tqdm import tqdm
data = pd.read_excel("/home/p1/Desktop/data/L1300-FDA-approved-Drug-Library-96-well.xlsx")
real_data = pd.read_excel("/home/p1/Desktop/data/connect1.xlsx")
real_data = real_data[real_data["CAS"].isin(data["CAS Number"].unique())]
# data = data[data["CAS Number"].isin(real_data["CAS"].unique())]
indications = data["Indication"].to_numpy()
indication_set = set()
for item in indications:
    # if "/" in item:
    #     item = item.split("/")[0] + "_" + item.split("/")[1]
    # for i in item.split("/"):
    #     indication_set.add(i)
    indication_set.add(item)
indication_set = list(indication_set)
indication_set = sorted(indication_set)
print(indication_set)
indication_dict = {item:i for i, item in enumerate(indication_set)}
def pro_row(row, pbar):
    pbar.update(1)
    cat = row["CAS"]
    cur_cat_data = data[data["CAS Number"]==cat]
    cur_indications = cur_cat_data["Indication"].to_numpy()

    # res = [0] * len(indication_set)
    label = indication_dict[cur_indications[0]]
    # if "/" in cur_indications[0]:
    #     item = cur_indications[0].split("/")[0] + "_" + cur_indications[0].split("/")[1]
    #     res[indication_dict[item]] = 1
    # else:
    #     for item in cur_indications[0].split("/"):
    #         res[indication_dict[item]] = 1
    return label
pbar = tqdm(total=real_data.shape[0])
real_data["label"] = real_data.apply(pro_row, axis=1, args=(pbar, ))

indication_res = copy.deepcopy(indication_set)
for item in indication_set:
    label_sum = (real_data["label"]==item).sum()
    if label_sum==10:
        indication_res.remove(item)
print(indication_res)
real_data.to_excel("/home/p1/Desktop/data/connect2.xlsx", index=False)

