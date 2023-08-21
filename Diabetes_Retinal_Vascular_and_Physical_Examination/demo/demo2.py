import pandas as pd

def add_row_to_excel(file_path, new_row):
    # 读取现有Excel文件
    df = pd.read_excel(file_path)

    # 将新行添加到DataFrame
    df.loc[len(df)] = new_row

    # 保存更新后的DataFrame回Excel文件
    df.to_excel(file_path, index=False)
    print(f"New row added to {file_path}")

# 示例使用
file_path = "D:/Analysis of diabetes/Test_8/预测模型结果.xlsx"
new_row = [1, 'John Doe', 25]  # 根据你的实际列结构调整
add_row_to_excel(file_path, new_row)
