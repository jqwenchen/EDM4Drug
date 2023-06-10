import pandas as pd
data_path = "/home/p1/Desktop/data/connect.xlsx"
xyz_data_path = "/home/p1/Desktop/data/230605_FDA_Compound_Library.xyz"
data_df = pd.read_excel(data_path)
with open(xyz_data_path, 'r') as f:
    content = f.read()
    contact = content.split('\n')
    cur_file_name = ""
    i = -1
    for line in contact:
        line = line.strip()
        if line == '' or line.isdigit():
            continue
        if line.endswith("cdx") or "maegz" in line:
            i += 1
            cur_file_name = data_df.iloc[i]["Entry ID"]
            continue
        with open(f"/home/p1/Desktop/data/xyz/{cur_file_name}.xyz", "a+") as f:
            f.write(line)
            f.write("\n")

