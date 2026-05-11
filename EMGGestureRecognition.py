import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
from tqdm import tqdm


WINDOW_SIZE = 1000   # ~1秒
STEP_SIZE   = 500   # 50%重叠
CHANNELS    = [f'channel{i}' for i in range(1, 9)]


# 单窗口进行特征提取
def extract_emg_features(window: np.ndarray) -> np.ndarray:
    feats = []
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        abs_sig = np.abs(sig)

        mav   = np.mean(abs_sig)                        # 平均绝对值
        rms   = np.sqrt(np.mean(sig ** 2))              # 均方根
        std   = np.std(sig)
        wl    = np.sum(np.abs(np.diff(sig)))            # 波长
        zc    = np.sum(np.diff(np.sign(sig)) != 0)     # 过零率
        ssc   = np.sum(np.diff(np.sign(np.diff(sig))) != 0)  # 斜率符号变化
        mx    = np.max(abs_sig)
        skew  = stats.skew(sig)
        kurt  = stats.kurtosis(sig)
        p25, p75 = np.percentile(sig, [25, 75])
        iqr   = p75 - p25
        energy = np.sum(sig ** 2)

        fft_vals = np.abs(np.fft.rfft(sig))
        freqs    = np.fft.rfftfreq(len(sig), d=1/1000)
        total_power = np.sum(fft_vals ** 2) + 1e-10
        mean_freq  = np.sum(freqs * fft_vals ** 2) / total_power
        median_freq = freqs[np.searchsorted(np.cumsum(fft_vals ** 2), total_power / 2)]
        low_power   = np.sum(fft_vals[(freqs >= 20)  & (freqs < 100)] ** 2) / total_power
        mid_power   = np.sum(fft_vals[(freqs >= 100) & (freqs < 300)] ** 2) / total_power
        high_power  = np.sum(fft_vals[(freqs >= 300) & (freqs < 500)] ** 2) / total_power

        feats.extend([mav, rms, std, wl, zc, ssc, mx, skew, kurt, iqr, energy,mean_freq, median_freq, low_power, mid_power, high_power])

    return np.array(feats)


# 对单个文件做窗口化特征提取
def process_file(file_path: str):
    df = pd.read_csv(file_path, sep='\t')
    df = df.drop(columns=['time'])

    X = df[CHANNELS].values
    y = df['class'].values
    features, labels = [], []

    for start in range(0, len(X) - WINDOW_SIZE + 1, STEP_SIZE):
        window  = X[start: start + WINDOW_SIZE]
        # 用窗口中出现次数最多的class作为标签
        label   = stats.mode(y[start: start + WINDOW_SIZE], keepdims=True).mode[0]
        features.append(extract_emg_features(window))
        labels.append(label)

    return np.array(features), np.array(labels)


# 按文件夹加载，逐文件处理
def load_data_by_folder(base_path: str):
    folders = sorted([
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ])

    train_folders = folders[:25]
    val_folders   = folders[25:]

    def load_folders(folder_list):
        all_X, all_y = [], []
        for folder in folder_list:
            folder_path = os.path.join(base_path, folder)
            for file in tqdm(os.listdir(folder_path), desc=folder):
                if not file.endswith('.txt'):
                    continue
                X, y = process_file(os.path.join(folder_path, file))
                all_X.append(X)
                all_y.append(y)
        return np.vstack(all_X), np.concatenate(all_y)

    print("处理训练集...")
    X_train, y_train = load_folders(train_folders)
    print("处理验证集...")
    X_val, y_val = load_folders(val_folders)

    return X_train, y_train, X_val, y_val


# 主流程
def main():

    base_path = "EMG_data_for_gestures-master"
    X_train, y_train, X_val, y_val = load_data_by_folder(base_path)

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"标签分布（训练）: {np.bincount(y_train.astype(int))}")
    print("训练集各类数量：",np.bincount(y_train.astype(int)))
    print("验证集各类数量：",np.bincount(y_val.astype(int)))

    # 去掉 class=0 和 7
    mask_train = (y_train != 0) &(y_train !=7)
    mask_val   = (y_val   != 0) &(y_val !=7)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_val,   y_val   = X_val[mask_val],     y_val[mask_val]
    print(f"过滤class=0和7后 — 训练: {X_train.shape}, 验证: {X_val.shape}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    print("训练 RandomForest...")
    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=8,       
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print("\n准确率：", accuracy_score(y_val, y_pred))
    print("\n分类报告：")
    print(classification_report(y_val, y_pred))
    

if __name__ == "__main__":
    main()

