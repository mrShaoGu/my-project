import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def conv_bn_relu(filters, kernel_size, strides=1):
    return tf.keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

class BasicBlock(layers.Layer):
    def __init__(self, filters, strides=1, downsample=None):
        super().__init__()
        self.conv1 = conv_bn_relu(filters, 3, strides)
        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])
        self.downsample = downsample
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        identity = x
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        if self.downsample is not None:
            identity = self.downsample(x, training=training)
        out += identity
        out = self.relu(out)
        return out

def make_layer(filters, blocks, strides):
    downsample = None
    if strides != 1:
        downsample = tf.keras.Sequential([
            layers.Conv2D(filters, 1, strides=strides, use_bias=False),
            layers.BatchNormalization()
        ])
    layers_list = [BasicBlock(filters, strides, downsample)]
    for _ in range(1, blocks):
        layers_list.append(BasicBlock(filters))
    return tf.keras.Sequential(layers_list)

class ResNet18(Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = conv_bn_relu(64, 3, 1)  # 适配MNIST，kernel=3, stride=1
        self.layer1 = make_layer(64, 2, 1)
        self.layer2 = make_layer(128, 2, 2)
        self.layer3 = make_layer(256, 2, 2)
        self.layer4 = make_layer(512, 2, 2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, x, training=False, return_feature=False):
        x = self.conv1(x, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        feat = self.avgpool(x)
        logits = self.fc(feat)
        if return_feature:
            return logits, feat
        return logits

# 1. 加载MNIST数据
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
y_train = y_train.astype(np.int32)
num_classes = 10

# 2. 定义ResNet-18（可用前述ResNet18类，略去此处，可用CIFAR版或自定义）
# 这里只给出特征提取部分，假设model为ResNet18实例
# model = ResNet18(num_classes=num_classes)
# model.build(input_shape=(None, 28, 28, 1))

# 3. 提取最后一层特征
all_features = []
all_labels = []
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
for batch_x, batch_y in train_dataset:
    # 获取全局池化后的特征
    features = model.feature_extractor(batch_x, training=False)
    all_features.append(features.numpy())
    all_labels.append(batch_y.numpy())
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 4. 计算全局均值和各类别均值
global_mean = np.mean(all_features, axis=0)
class_means = []
for c in range(num_classes):
    class_feat = all_features[all_labels == c]
    class_mean = np.mean(class_feat, axis=0)
    class_means.append(class_mean)
class_means = np.stack(class_means, axis=0)

print("Global mean shape:", global_mean.shape)
print("Class means shape:", class_means.shape)

import numpy as np

# 假设你已经有如下变量
# global_mean: shape (feature_dim,)
# class_means: shape (num_classes, feature_dim)
# all_features: shape (N, feature_dim)
# all_labels: shape (N,)
# model: 训练好的ResNet18模型

num_classes = class_means.shape[0]
feature_dim = class_means.shape[1]

# 1. Within-class Variability Collapse
within_class_cov = []
for c in range(num_classes):
    class_feat = all_features[all_labels == c]
    class_mean = class_means[c]
    centered = class_feat - class_mean
    cov = np.matmul(centered.T, centered) / (len(class_feat) - 1)
    within_class_cov.append(cov)
within_class_cov = np.stack(within_class_cov, axis=0)
within_class_var = np.mean([np.trace(cov) for cov in within_class_cov])
print("Average within-class covariance trace:", within_class_var)

# 2. Convergence of the Learned Classifier to a Simplex ETF
def get_simple_etf(d, c):
    I = np.eye(c)
    ones = np.ones((c, c)) / c
    M = I - ones
    U, _, _ = np.linalg.svd(M)
    etf = U[:, :d].T * np.sqrt(c / (c - 1))
    return etf.astype(np.float32)  # shape: [d, c]

# 提取权重
W = model.fc.weights[0].numpy()  # shape: [feature_dim, num_classes] or [num_classes, feature_dim]
if W.shape[0] == num_classes:
    W = W.T  # [feature_dim, num_classes]
W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)
etf = get_simple_etf(feature_dim, num_classes)
etf_norm = etf / np.linalg.norm(etf, axis=0, keepdims=True)
cos_sim = np.abs(np.sum(W_norm * etf_norm) / num_classes)
print("Cosine similarity between classifier and ETF:", cos_sim)

# 3. Convergence to Self-duality
class_means_norm = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
W_norm_T = W.T / np.linalg.norm(W.T, axis=1, keepdims=True)
cos_sim_matrix = np.matmul(class_means_norm, W_norm_T.T)
print("Class mean vs. classifier weight cosine similarity matrix:\n", cos_sim_matrix)
print("Diagonal mean:", np.mean(np.diag(cos_sim_matrix)))
print("Off-diagonal mean:", (np.sum(cos_sim_matrix) - np.trace(cos_sim_matrix)) / (num_classes**2 - num_classes))

# 4. Collapse of the Bias
bias = model.fc.bias.numpy()
print("Classifier bias:", bias)
print("Bias std:", np.std(bias))