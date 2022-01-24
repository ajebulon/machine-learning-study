import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
np.random.seed(0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(x, w, b):
    z = np.matmul(w, x) + b
    a = sigmoid(z)
    return a

def compute_cost(x, y, w, b):
    # Forward propagation
    y_hat = forward_prop(x, w, b)
    
    # Using cross-entropy loss function for classification use case
    N = y.shape[1]
    J = 1 / N * np.sum(-y*np.log(y_hat) - (1-y) * np.log(1-y_hat), axis=1)
    return J.item()
    

SAMPLE_NM = 1000
x = 10 * np.random.random(size=(2, SAMPLE_NM))
y = np.zeros(shape=(1, SAMPLE_NM))
y[:, x[0, :] > x[1, :]] = 1

# plt.figure()
# plt.scatter(x[0, :], x[1, :])
# plt.plot(y.T)

plt.figure()
plt.grid()
plt.title("Decision boundary target")
plt.scatter(x[0, np.argwhere(y == 0)[:, 1]], x[1, np.argwhere(y == 0)[:, 1]], marker='x', c='red')
plt.scatter(x[0, np.argwhere(y == 1)[:, 1]], x[1, np.argwhere(y == 1)[:, 1]], marker='x', c='green')


# Initialization
w = np.random.random(size=(1, 2)) - 0.5
b = np.random.random(size=(1, 1)) - 0.5


# Initial performance
y_hat = forward_prop(x, w, b)
y_out = np.zeros(shape=y_hat.shape)
y_out[0, y_hat[0, :] >= 0.5] = 1
y_out[0, y_hat[0, :] < 0.5] = 0

plt.figure()
plt.grid()
plt.title("Decision boundary before training")
plt.scatter(x[0, np.argwhere(y_out == 0)[:, 1]], x[1, np.argwhere(y_out == 0)[:, 1]], marker='x', c='red')
plt.scatter(x[0, np.argwhere(y_out == 1)[:, 1]], x[1, np.argwhere(y_out == 1)[:, 1]], marker='x', c='green')

accuracy = np.mean(y_out == y) * 100
print("Initial accuracy: {:.2f}%".format(accuracy))


# Training
EPOCH_NM = 10000
cost_history = [compute_cost(x, y, w, b)]

for epoch in range(EPOCH_NM):

    # Forward propagation
    y_hat = forward_prop(x, w, b)
    
    # Backward propagation
    N = y.shape[1]
    dJw = 1 / N * np.matmul(y_hat - y, x.T)
    dJb = 1 / N * np.sum(y_hat - y)
    
    # Update weight
    L_RATE = 1e-3
    w = w - L_RATE * dJw
    b = b - L_RATE * dJb
    
    # Check cost
    cost = compute_cost(x, y, w, b)
    cost_history.append(cost)
    
print("Initial cost: {:.6f}".format(cost_history[0]))
print("Final cost: {:.6f}".format(cost_history[-1]))

plt.figure()
plt.grid()
plt.plot(cost_history)


# Final performance
y_hat = forward_prop(x, w, b)
y_out = np.zeros(shape=y_hat.shape)
y_out[0, y_hat[0, :] >= 0.5] = 1
y_out[0, y_hat[0, :] < 0.5] = 0

plt.figure()
plt.grid()
plt.title("Decision boundary after training")
plt.scatter(x[0, np.argwhere(y_out == 0)[:, 1]], x[1, np.argwhere(y_out == 0)[:, 1]], marker='x', c='red')
plt.scatter(x[0, np.argwhere(y_out == 1)[:, 1]], x[1, np.argwhere(y_out == 1)[:, 1]], marker='x', c='green')

accuracy = np.mean(y_out == y) * 100
print("Final accuracy: {:.2f}%".format(accuracy))
