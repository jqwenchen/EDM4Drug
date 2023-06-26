import pandas as pd
from tqdm import tqdm
data_path = "/home/p1/Desktop/0615_L1700_Compound Library/信息对应表.xlsx"
xyz_data_path = "/home/p1/Desktop/0615_L1700_Compound Library/Structures_L1700.xyz"
data_df = pd.read_excel(data_path)
with open(xyz_data_path, 'r') as f:
    content = f.read()
    contact = content.split('\n')
    cur_file_name = ""
    i = -1
    is_id = False
    for line in tqdm(contact):
        line = line.strip()
        if line == '' or line.isdigit():
            is_id = True
            continue
        # if line.endswith("cdx") or "maegz" in line:
        if is_id:
            i += 1
            cur_file_name = data_df.iloc[i]["Entry ID"]
            is_id = False
            continue
        with open(f"/home/p1/Desktop/0615_L1700_Compound Library/xyz/{cur_file_name}.xyz", "a+") as f:
            f.write(line)
            f.write("\n")

