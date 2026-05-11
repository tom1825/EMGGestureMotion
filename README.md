# EMG Gesture Recognition

## 一、项目简介
基于八通道表面肌电信号（sEMG）的手势识别项目。通过对原始 EMG 信号进行滑动窗口分割和多维特征提取，结合机器学习分类器实现对 6 种手势动作的自动识别。

## 二、数据集
- **来源：** （[EMG data for gestures](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures)）
- **结构：** 32 个文件夹，每个文件夹对应一名受试者的一次采集，每个文件夹包含 2 个 `.txt` 文件
- **信号：** 8 通道表面肌电信号，幅值量级约 1e-5,每个手势执行3秒，手势之间停顿3秒。
- **标签：** class 0-7（7 种手势），其中 class 7 仅存在于第 11、30 号文件夹
  ```
  0 - unmarked data, 
  1 - hand at rest, 
  2 - hand clenched in a fist, 
  3 - wrist flexion, 
  4 - wrist extension, 
  5 - radial deviations, 
  6 - ulnar deviations, 
  7 - extended palm (the gesture was not performed by all subjects).
  ```
- **划分：** 前 25 个文件夹为训练集，后 7 个文件夹为验证集（第 30 号文件夹强制划入训练集以保证 class 7 的跨人覆盖）

## 三、任务目标

对每个时间窗口的 EMG 信号进行多分类，预测当前窗口对应的手势类别（class 1-7）。

## 四、处理流程

### 1. 数据加载与数量统计
按文件夹划分训练/验证集，逐文件独立处理，避免跨文件边界污染。
经过`count.py`统计：
```
{0: 2440245, 1: 222326, 2: 217079, 3: 223593, 4: 224732, 5: 225158, 6: 226516, 7: 13696}
```
class = 7 只出现在第11和第30个文件夹中，且严重不平衡(约其他类别的5%)，若加入训练，会极大影响准确率，因此暂且当作噪声处理，待后续解决。

### 2. 滑动窗口
- 窗口大小：1000 个采样点（约 1 秒）
- 步长：500 个采样点（50% 重叠）
- 窗口标签：取窗口内出现次数最多的 class（众数投票）

### 3. 特征提取
对每个通道提取 16 维特征(包括时域幅值、时域形态、统计分布和频域)
```
mav, rms, std, wl, zc, ssc, mx, skew, kurt, iqr, energy,mean_freq, median_freq, low_power, mid_power, high_power
````
8 通道共 128 维：

### 4. 预处理
- 过滤 class=0 静息段和class=7
- StandardScaler 标准化

**5. 模型训练**
* 使用 RandomForestClassifier，启用 early stopping，以验证集 multi_logloss 为监控指标。
* 使用：
```python
min_samples_leaf=2,
max_features='sqrt',
```
防止节点过度分裂导致的过拟合        

## 五、评估指标
- **Accuracy**：整体分类准确率
- **Precision / Recall / F1-score**：各类别单独评估
- **Classification Report**：输出每个类别的详细指标及宏平均、加权平均

## 六、实验结果
```
准确率： 0.8571428571428571

分类报告：
              precision    recall  f1-score   support

           1       0.99      0.84      0.91        98
           2       0.96      0.91      0.93        93
           3       0.78      0.90      0.84        97
           4       0.83      0.88      0.85        97
           5       0.79      0.82      0.80        94
           6       0.84      0.80      0.82        95

    accuracy                           0.86       574
   macro avg       0.86      0.86      0.86       574
weighted avg       0.86      0.86      0.86       574
```

## 七、核心结论
### 1. 数据划分是最关键的因素。
class 7 仅分布在两个文件夹且来自不同受试者，若训练集只覆盖其中一人，模型完全无法跨人泛化，f1 直接归零。合理的数据划分比算法选择影响更大。
### 2. 特征工程是主要提升来源。
引入频域特征（平均频率、频带能量）和 EMG 专用时域特征（WL、ZC、SSC）后，相比仅使用均值/标准差的基线有明显提升。
### 3. 各类样本数量差异较大。
class=7数量太少，不足以让模型学习，保留只会成为噪声，做出删除处理准确率提高较大。但是会存在以后class=7类别都无法预测的问题。
### 4. 个体差异是 EMG 识别的核心难题。
不同受试者的肌肉解剖结构和发力习惯不同，跨人泛化能力是实际部署中最需要解决的问题。
