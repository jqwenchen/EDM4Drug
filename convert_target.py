import copy
import pandas as pd
from tqdm import tqdm
import numpy as np

data = pd.read_excel("/home/p1/Desktop/0615_L1700_Compound Library/L1700-II-Bioactive-Compound-Library-II-(Provided-by-Pfizer).xlsx")
real_data = pd.read_excel("/home/p1/Desktop/0615_L1700_Compound Library/connect1.xlsx")
smile_data = pd.read_excel("/home/p1/Desktop/0615_L1700_Compound Library/Smiles_L1700.xlsx")
# real_data = real_data[real_data["CAS"].isin(data["CAS Number"].unique())]
# indications = data["Indication"].to_numpy()
real_data = real_data[real_data["Title"].isin(data["COMPOUND_NAME"].unique())]
indications = data["pathway"].to_numpy()
indication_set = set()
for item in indications:
    if type(item) == str or not np.isnan(item):
        indication_set.add(item.strip().replace(",", "").replace(" ", "").replace("\n", "").replace("，", "").replace("&", ""))
indication_set = list(indication_set)
indication_set = sorted(indication_set)
print(indication_set)
indication_dict = {item:i for i, item in enumerate(indication_set)}

def pro_row(row, pbar):
    pbar.update(1)
    # cat = row["CAS"]
    # cur_cat_data = data[data["CAS Number"]==cat]
    # cur_indications = cur_cat_data["Indication"].to_numpy()
    cat = row["Title"]
    cur_cat_data = data[data["COMPOUND_NAME"] == cat]
    cur_indications = cur_cat_data["pathway"].to_numpy()
    if type(cur_indications[0]) != str and np.isnan(cur_indications[0]):
        return -1, ""
    label = indication_dict[cur_indications[0].strip().replace(",", "").replace(" ", "").replace("\n", "").replace("，", "").replace("&", "")]
    smile = smile_data[smile_data["Column2"]==cat]["Column1"]
    if len(smile)==0:
        return -1, ""
    return label, smile.tolist()[0]

pbar = tqdm(total=real_data.shape[0])
real_data[["label","smiles"]] = real_data.apply(pro_row, axis=1, args=(pbar, ), result_type="expand")
real_data = real_data[real_data["label"]!=-1]

# indication_res = copy.deepcopy(indication_set)
# for item in indication_set:
#     label_sum = (real_data["label"]==item).sum()
#     if label_sum==10:
#         indication_res.remove(item)
# print(indication_res)




real_data.to_excel("/home/p1/Desktop/0615_L1700_Compound Library/connect2.xlsx", index=False)

