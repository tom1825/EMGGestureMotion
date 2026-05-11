import os
import pandas as pd


def count_class_in_file(file_path):
    df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    counts = df['class'].value_counts().to_dict()

    return {i: counts.get(i, 0) for i in range(8)}


def main():
    root_path = "EMG_data_for_gestures-master"
    total = {i: 0 for i in range(8)}

    print("\n每个文件夹统计\n")

    # 遍历32个文件夹
    for folder in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder)

        if not os.path.isdir(folder_path):
            continue
        folder_count = {i: 0 for i in range(8)}

        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                try:
                    c = count_class_in_file(file_path)

                    for k in range(8):
                        folder_count[k] += c[k]
                        total[k] += c[k]
                except Exception as e:
                    print("跳过:", file_path, e)

        print(f"Folder {folder}: {folder_count}")

    print("\n全局统计（32个文件夹）\n")
    print(total)


if __name__ == "__main__":
    main()