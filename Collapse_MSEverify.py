import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time


# ResNet18定义
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
        self.conv1 = conv_bn_relu(64, 3, 1)
        self.layer1 = make_layer(64, 2, 1)
        self.layer2 = make_layer(128, 2, 2)
        self.layer3 = make_layer(256, 2, 2)
        self.layer4 = make_layer(512, 2, 2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, x, training=False, return_feature=False):
        x = tf.reshape(x, [-1, 32, 32, 3])  # 适配CIFAR10
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


def get_simple_etf(d, c):
    I = np.eye(c)
    ones = np.ones((c, c)) / c
    M = I - ones
    U, _, _ = np.linalg.svd(M)
    etf = U[:, :d].T * np.sqrt(c / (c - 1))
    return etf.astype(np.float32)  # shape: [d, c]


# 数据准备
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = y_train.flatten().astype(np.int32)
y_test = y_test.flatten().astype(np.int32)
output_size = 10
batch_size = 64
epochs = 5

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# 构建ResNet18模型
model = ResNet18(num_classes=output_size)
_ = model(x_train[:1])  # build

loss_fn = losses.MeanSquaredError()  # 损失函数改为MSE
optimizer = optimizers.Adam(learning_rate=0.001)
weight_decay = 1e-4

# 指标记录
train_accs, test_accs = [], []
train_f1s, test_f1s = [], []
train_aucs, test_aucs = [], []
nc1_list, nc2_list, nc3_list, nc4_list = [], [], [], []
epoch_time_list = []

for epoch in range(epochs):
    start_time = time.time()
    # 训练
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits, feats = model(x_batch, training=True, return_feature=True)
            y_onehot = tf.one_hot(y_batch, depth=output_size)
            mse_loss = loss_fn(y_onehot, logits)  # 损失函数为MSE
            l2_loss = tf.nn.l2_loss(model.fc.kernel) + tf.nn.l2_loss(model.fc.bias) + tf.nn.l2_loss(feats)
            loss = mse_loss + weight_decay * l2_loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 评估
    def eval_model(dataset):
        y_true, y_pred, y_prob, features = [], [], [], []
        for x_batch, y_batch in dataset:
            logits, feats = model(x_batch, training=False, return_feature=True)
            preds = tf.argmax(logits, axis=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
            y_prob.extend(tf.nn.softmax(logits).numpy())
            features.append(feats.numpy())
        return (np.array(y_true), np.array(y_pred), np.array(y_prob), np.concatenate(features, axis=0))

    y_true_train, y_pred_train, y_prob_train, feats_train = eval_model(train_dataset)
    y_true_test, y_pred_test, y_prob_test, feats_test = eval_model(test_dataset)

    train_acc = np.mean(y_pred_train == y_true_train)
    test_acc = np.mean(y_pred_test == y_true_test)
    train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
    test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
    try:
        train_auc = roc_auc_score(np.eye(output_size)[y_true_train], y_prob_train, multi_class='ovr')
        test_auc = roc_auc_score(np.eye(output_size)[y_true_test], y_prob_test, multi_class='ovr')
    except:
        train_auc = test_auc = np.nan

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)
    train_aucs.append(train_auc)
    test_aucs.append(test_auc)

    # NC指标
    features = feats_train
    labels = y_true_train
    class_means = np.stack([features[labels == c].mean(axis=0) for c in range(output_size)], axis=0)
    global_mean = features.mean(axis=0)
    class_means_centered = class_means - global_mean
    within_cov = sum(np.cov((features[labels == c] - class_means[c]).T) * (labels == c).sum() for c in range(output_size)) / len(features)
    between_cov = sum(np.outer(class_means_centered[c], class_means_centered[c]) * (labels == c).sum() for c in range(output_size)) / len(features)
    nc1 = np.trace(within_cov @ np.linalg.pinv(between_cov)) / output_size if np.linalg.norm(between_cov) > 1e-8 else 0.0
    fc_weights = model.fc.kernel.numpy().T
    etf = get_simple_etf(fc_weights.shape[1], output_size)
    nc2 = np.linalg.norm(fc_weights @ fc_weights.T / np.trace(fc_weights @ fc_weights.T) - (np.eye(output_size) - np.ones((output_size, output_size)) / output_size))
    cos_sim = [np.dot(class_means[i], fc_weights[i]) / (np.linalg.norm(class_means[i]) * np.linalg.norm(fc_weights[i])) for i in range(output_size)]
    nc3 = 1 - np.mean(cos_sim)
    fc_bias = model.fc.bias.numpy()
    nc4 = np.linalg.norm(fc_bias + np.dot(fc_weights, global_mean))
    nc1_list.append(nc1)
    nc2_list.append(nc2)
    nc3_list.append(nc3)
    nc4_list.append(nc4)

    epoch_time_list.append(time.time() - start_time)
    print(f"Epoch {epoch+1}/{epochs} finished. Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

# 绘图
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.legend(); plt.title("Accuracy"); plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(train_f1s, label="Train F1")
plt.plot(test_f1s, label="Test F1")
plt.legend(); plt.title("F1 Score"); plt.grid(True)
plt.subplot(1, 3, 3)
plt.plot(train_aucs, label="Train AUC")
plt.plot(test_aucs, label="Test AUC")
plt.legend(); plt.title("AUC"); plt.grid(True)
plt.tight_layout(); plt.show()

plt.figure(figsize=(15, 4))
for i, (nc, name) in enumerate(zip([nc1_list, nc2_list, nc3_list, nc4_list], ['NC1','NC2','NC3','NC4'])):
    plt.subplot(1, 4, i+1)
    plt.plot(nc)
    plt.title(name)
    plt.grid(True)
plt.tight_layout(); plt.show()