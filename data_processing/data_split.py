import pandas as pd

data_path = "/home/chenye/project/RibonanzaNet/data/test_sequences.csv"
save_path = "/home/chenye/project/RibonanzaNet/data/chenye_try.csv"
number_row = 1000

df = pd.read_csv(data_path, nrows=number_row)

df.to_csv(save_path,index=False)