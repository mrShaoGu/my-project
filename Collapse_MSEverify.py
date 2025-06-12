import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import time

# 1. 定义模型
class SimpleNeuralNetworkModel(Model):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc1 = layers.Dense(hidden_size1, activation='relu')
        self.fc2 = layers.Dense(hidden_size2, activation='relu')
        self.fc3 = layers.Dense(output_size)
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 2. 超参数
input_dim = 32
hidden_size1 = 64
output_size = 10
hidden_size2 = output_size  # 让最后一层特征维度等于类别数
batch_size = 32
epochs = 10 # 增加训练轮数以观察神经崩溃现象

# 3. 生成MNIST数据
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 28*28)  # 展平成向量
y_train = y_train.astype(np.int32)
input_dim = 28 * 28  # 更新输入维度

# 如果需要使用MNIST数据集，请取消上面的注释并注释掉下面的CIFAR-10部分

# # 3. 生成CIFAR-10数据
# (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
# x_train = x_train.astype(np.float32) / 255.0
# x_train = x_train.reshape(-1, 32*32*3)  # 展平成向量
# y_train = y_train.astype(np.int32).flatten()
# input_dim = 32 * 32 * 3  # CIFAR-10输入维度

# 划分训练集和测试集
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
train_data = tf.convert_to_tensor(x_train_split)
train_labels = tf.convert_to_tensor(y_train_split)
test_data = tf.convert_to_tensor(x_test_split)
test_labels = tf.convert_to_tensor(y_test_split)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

# 4. 构建模型
model = SimpleNeuralNetworkModel(input_dim, hidden_size1, hidden_size2, output_size)

# 先让模型跑一次，build权重
_ = model(tf.convert_to_tensor(x_train_split[:1]))

def get_simple_etf(d, c):
    # d: 特征维度, c: 类别数
    I = np.eye(c)
    ones = np.ones((c, c)) / c
    M = I - ones
    U, S, Vt = np.linalg.svd(M)
    etf = U[:, :d].T * np.sqrt(c / (c - 1))
    return etf.astype(np.float32)  # shape: [d, c]

# 设置fc3权重为ETF并冻结
etf_weight = get_simple_etf(hidden_size2, output_size).T  # shape: [output_size, hidden_size2]
model.fc3.kernel.assign(etf_weight)
model.fc3.kernel._trainable = False
model.fc3.bias._trainable = True

# 5. 损失函数和优化器
loss_fn = losses.MeanSquaredError()  # ← 改为交叉熵
optimizer = optimizers.Adam(learning_rate=0.001)  # 可根据需要改为Adam或SGD
weight_decay = 1e-4 

# 6. 训练模型并记录accuracy、F1、AUC
train_accuracy_list = []
test_accuracy_list = []
train_f1_list = []
test_f1_list = []
train_auc_list = []
test_auc_list = []
epoch_time_list = []  # 新增：记录每个epoch的耗时

for epoch in range(epochs):
    start_time = time.time()  # 新增：记录epoch开始时间

    # 训练
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            h1 = model.fc1(x_batch)
            h2 = model.fc2(h1)
            logits = model.fc3(h2)
            y_batch_onehot = tf.one_hot(y_batch, depth=output_size)
            ce_loss = loss_fn(y_batch_onehot, logits)
            l2_fc3_kernel = tf.nn.l2_loss(model.fc3.kernel)
            l2_fc3_bias = tf.nn.l2_loss(model.fc3.bias)
            l2_h2 = tf.nn.l2_loss(h2)
            loss = ce_loss + weight_decay * (l2_fc3_kernel + l2_fc3_bias + l2_h2)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # 训练集指标
    y_true_train = []
    y_pred_train = []
    y_prob_train = []
    for x_batch, y_batch in train_dataset:
        logits = model(x_batch)
        preds = tf.argmax(logits, axis=1)
        y_true_train.extend(y_batch.numpy())
        y_pred_train.extend(preds.numpy())
        y_prob_train.extend(tf.nn.softmax(logits).numpy())
    train_acc = np.mean(np.array(y_pred_train) == np.array(y_true_train))
    train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
    try:
        train_auc = roc_auc_score(
            np.eye(output_size)[y_true_train], np.array(y_prob_train), multi_class='ovr'
        )
    except:
        train_auc = np.nan
    train_accuracy_list.append(train_acc)
    train_f1_list.append(train_f1)
    train_auc_list.append(train_auc)

    # 测试集指标
    y_true_test = []
    y_pred_test = []
    y_prob_test = []
    for x_batch, y_batch in test_dataset:
        logits = model(x_batch)
        preds = tf.argmax(logits, axis=1)
        y_true_test.extend(y_batch.numpy())
        y_pred_test.extend(preds.numpy())
        y_prob_test.extend(tf.nn.softmax(logits).numpy())
    test_acc = np.mean(np.array(y_pred_test) == np.array(y_true_test))
    test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
    try:
        test_auc = roc_auc_score(
            np.eye(output_size)[y_true_test], np.array(y_prob_test), multi_class='ovr'
        )
    except:
        test_auc = np.nan
    test_accuracy_list.append(test_acc)
    test_f1_list.append(test_f1)
    test_auc_list.append(test_auc)

    end_time = time.time()  # 新增：记录epoch结束时间
    epoch_duration = end_time - start_time
    epoch_time_list.append(epoch_duration)
    print(f"Epoch {epoch+1}/{epochs} finished in {epoch_duration:.2f} seconds.")

# 显示平均每个epoch耗时
avg_time = np.mean(epoch_time_list)
print(f"Average time per epoch: {avg_time:.2f} seconds.")

# 绘制 accuracy, F1, AUC 曲线
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs+1), train_accuracy_list, label="Train Accuracy")
plt.plot(range(1, epochs+1), test_accuracy_list, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, epochs+1), train_f1_list, label="Train F1")
plt.plot(range(1, epochs+1), test_f1_list, label="Test F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Epoch")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, epochs+1), train_auc_list, label="Train AUC")
plt.plot(range(1, epochs+1), test_auc_list, label="Test AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("AUC vs. Epoch")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ========== Neural Collapse 验证部分 ==========

# 提取最后一层特征和标签
features = []
labels = []
for x_batch, y_batch in train_dataset:
    feats = model.fc2(model.fc1(x_batch))  # 取倒数第二层输出
    features.append(feats.numpy())
    labels.append(y_batch.numpy())
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# 计算每个类别的特征均值
class_means = []
for c in range(output_size):
    class_means.append(features[labels == c].mean(axis=0))
class_means = np.stack(class_means, axis=0)

# 1. 检查类均值之间的夹角（正交性和等距性）
sim = cosine_similarity(class_means)
plt.figure(figsize=(6,5))
plt.imshow(sim, cmap='viridis')
plt.colorbar()
plt.title("Class Mean Cosine Similarity Matrix")
plt.xlabel("Class")
plt.ylabel("Class")
plt.show()

# 2. 检查特征均值与分类器权重的对齐
fc3_weights = model.fc3.kernel.numpy().T  # shape: [output_size, feature_dim]
cos_list = []
for i in range(output_size):
    cos = np.dot(class_means[i], fc3_weights[i]) / (np.linalg.norm(class_means[i]) * np.linalg.norm(fc3_weights[i]))
    cos_list.append(cos)
plt.figure()
plt.bar(range(output_size), cos_list)
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Cosine Similarity")
plt.title("Class Mean vs. Weight Cosine Similarity")
plt.show()

# 3. 检查类内协方差
cov_means = []
for c in range(output_size):
    feats_c = features[labels == c]
    cov = np.cov(feats_c, rowvar=False)
    cov_means.append(np.mean(np.abs(cov)))
plt.figure()
plt.bar(range(output_size), cov_means)
plt.xlabel("Class")
plt.ylabel("Mean Abs Covariance")
plt.title("Class-wise Feature Covariance Mean")
plt.show()