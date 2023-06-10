import pandas as pd
file_path = "/home/p1/Desktop/data/connect.xlsx"
connect_data = pd.read_excel(file_path, sheet_name="Sheet2")
info_data = pd.read_excel(file_path, sheet_name="Sheet1")
