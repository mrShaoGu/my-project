import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import matplotlib.pyplot as plt

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
hidden_size2 = 128
output_size = 10
batch_size = 32
epochs = 300

# 3. 生成模拟数据
train_data = tf.random.normal([1000, input_dim])
train_labels = tf.random.uniform([1000], minval=0, maxval=output_size, dtype=tf.int32)
test_data = tf.random.normal([200, input_dim])
test_labels = tf.random.uniform([200], minval=0, maxval=output_size, dtype=tf.int32)

# 4. 构建模型
model = SimpleNeuralNetworkModel(input_dim, hidden_size1, hidden_size2, output_size)

# 5. 损失函数和优化器
# 损失函数改为均方误差（MSE）+ L2正则化
loss_fn = losses.MeanSquaredError()
optimizer = optimizers.SGD(learning_rate=0.001)
weight_decay = 1e-4  # 正则化系数，可根据需要调整

# 6. 训练循环
loss_history = []
accuracy_history = []
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(1000).batch(batch_size)
for epoch in range(epochs):
    # 训练
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch)
            # 将标签转换为one-hot编码以适配MSE损失
            y_batch_onehot = tf.one_hot(y_batch, depth=output_size)
            mse_loss = loss_fn(y_batch_onehot, logits)
            # L2正则化项
            fc3_bias = model.fc3.bias  # shape: [output_size]
            l2_bias_loss = tf.nn.l2_loss(fc3_bias)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
            loss = mse_loss + weight_decay * l2_loss+ weight_decay * l2_bias_loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_history.append(loss.numpy())

    # 每100次epoch输出一次loss
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    # 计算训练集 accuracy
    train_logits = model(train_data)
    train_pred = tf.argmax(train_logits, axis=1, output_type=tf.int32)
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(train_pred, train_labels), tf.float32))
    accuracy_history.append(train_accuracy.numpy())

# 绘制 loss 曲线
plt.figure()
plt.plot(range(1, epochs+1), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.show()

# 绘制 accuracy 曲线
plt.figure()
plt.plot(range(1, epochs+1), accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.show()

# 7. 测试模型
logits = model(test_data)
pred = tf.argmax(logits, axis=1, output_type=tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, test_labels), tf.float32))
print(f"Test Accuracy: {accuracy.numpy():.4f}")