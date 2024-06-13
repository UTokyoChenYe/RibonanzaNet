import pickle
import pandas as pd

# 定义保存的文件路径
preds_file_path = "../preds.p"
output_csv_path = "./preds.csv"

# 打开并读取文件
with open(preds_file_path, 'rb') as f:
    preds_dict = pickle.load(f)

# 将字典转换为 DataFrame
df = pd.DataFrame(list(preds_dict.items()), columns=['sequence_id', 'prediction'])

# 保存为 CSV 文件
df.to_csv(output_csv_path, index=False)

print(f"Predictions have been saved to {output_csv_path}")
