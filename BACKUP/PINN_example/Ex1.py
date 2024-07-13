import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)), #input node 하나
    tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units=1) # output node 하나
])
"""
print(NN.summary())
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 32)                64

 dense_1 (Dense)             (None, 32)                1056

 dense_2 (Dense)             (None, 32)                1056

 dense_3 (Dense)             (None, 1)                 33

=================================================================
Total params: 2,209
Trainable params: 2,209
Non-trainable params: 0
_________________________________________________________________
"""

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

def ode_system(t, net):
    t = t.reshape(-1, 1) # (첫번째 차원, 두번째 차원)
    t = tf.constant(t, dtype=tf.float32)
    t_0 = tf.zeros((1,1))
    one = tf.ones((1,1))

    #tensorflow에서는 gradient 계산한 다음에 버리는 식으로 되기 때문에
    #tf.GradientTape()를 tape로 붙여놔라 라는 뜻.
    with tf.GradientTape() as tape:
        tape.watch(t)
        u = net(t)
        u_t = tape.gradient(u,t)
    
    ode_loss = u_t - tf.math.cos(2*np.pi*t)
    IC_loss = net(t_0) - one

    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)

    return total_loss

train_t = (np.random.rand(30)*2).reshape(-1, 1)
train_loss_record = []

for itr in range(6000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 ==0:
        print(train_loss.numpy())
        #[1.4985945, 0.0019368896, 0.0011844431, 0.0005937301, 0.00016220543, 9.761003e-05]
# plt.figure(figsize=(10,8))
# plt.plot(train_loss_record)
# plt.show()

test_t = np.linspace(0,2,100)

train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
true_u = np.sin(2*np.pi*test_t)/(2*np.pi) +1

pred_u = NN.predict(test_t).ravel()

plt.figure(figsize=(10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k', label= 'True')
plt.plot(test_t, pred_u, '--r', label='Prediction')
plt.legend(fontsize=15)
plt.xlabel('t', fontsize = 5)
plt.ylabel('u', fontsize = 15)
plt.show()


           