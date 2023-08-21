import pandas as pd

def result_cast(input_file_path, sheet_name):
    column_replace_dict = {
        'Result':{1: 0, 2: 1}
    }

    df = pd.read_excel(input_file_path, sheet_name = sheet_name)

    for column, value_map in column_replace_dict.items():
        if column in df.columns:
            df[column] = df[column].replace(value_map)

    with pd.ExcelWriter(input_file_path, mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name,index=False)

if __name__ == '__main__':
    input_file_path = "D:/Analysis of diabetes/Test_5/data_combine.xlsx"
    sheet_name = "中间范围_糖尿病"
    result_cast(input_file_path, sheet_name)