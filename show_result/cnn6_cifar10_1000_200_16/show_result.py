import numpy as np
import matplotlib.pyplot as plt

# results_bo_emukit = np.load('./running_results_bo_emukit_with_constraint_15s.npy', allow_pickle=True)
# results_bo_emukit = np.delete(results_bo_emukit, 0, 0)
# best_result_bo_emukit = results_bo_emukit[results_bo_emukit[:, 0].argsort()][0]

results_bohb = np.load('./running_results_based_cnn6_cifar10_1000_200.npy', allow_pickle=True)
results_bohb = np.delete(results_bohb, 0, 0)
best_result_bohb = results_bohb[-1, :]

results_bohb_incremental = np.load('./running_results_inc_cnn6_cifar_1000_200.npy', allow_pickle=True)
results_bohb_incremental = np.delete(results_bohb_incremental, 0, 0)
best_result_bohb_incremental = results_bohb_incremental[-1, :]

# results_bohb_meta = np.load('./bohb_running_results_inc_and_meta_cnn_MNIST_5_0.5.npy', allow_pickle=True)
# results_bohb_meta = np.delete(results_bohb_meta, 0, 0)
# best_result_bohb_meta = results_bohb_meta[-1, :]
# print("best_result_bo_emukit:")
# print("loss: ", best_result_bo_emukit[0])
# print("test accurancy: ", best_result_bo_emukit[1])
# print("trining time:", best_result_bo_emukit[2])
# print("record time:", best_result_bo_emukit[3])
# print("config: ", best_result_bo_emukit[4])


o_results = np.load('./running_results_bo_cnn6_cifar10_1000_200.npy', allow_pickle=True)
o_re = np.delete(o_results, 0, 0)
n_results = np.load('./running_results_new_cnn6_cifar10_1000_200.npy', allow_pickle=True)
n_re = np.delete(n_results, 0, 0)
#print(results)
o_best_result = o_re[o_re[:, 0].argsort()][0]
print("original best result:")
print("loss: ", o_best_result[0])
print("test accurancy: ", o_best_result[1])
print("trining time:", o_best_result[2])
print("record time:", o_best_result[3])
print("config: ", o_best_result[4])

n_best_result = n_re[n_re[:, 0].argsort()][0]
print("best result:")
print("loss: ", n_best_result[0])
print("test accurancy: ", n_best_result[1])
print("trining time:", n_best_result[2])
print("record time:", n_best_result[3])
print("config: ", n_best_result[4])

o_x_axis = o_re[:, 3]
o_y_axis = o_re[:, 1]

n_x_axis = n_re[:, 3]
n_y_axis = n_re[:, 1]


print("best_result_bohb:")
print("loss: ", best_result_bohb[0])
print("test accurancy: ", best_result_bohb[2])
print("trining time:", best_result_bohb[3])
print("record time:", best_result_bohb[4])
print("size of model: ", best_result_bohb[5])
print("config: ", best_result_bohb[6])
print("budget:", best_result_bohb[7])

print("best_result_bohb_incremental:")
print("loss: ", best_result_bohb_incremental[0])
print("test accurancy: ", best_result_bohb_incremental[2])
print("trining time:", best_result_bohb_incremental[3])
print("record time:", best_result_bohb_incremental[4])
print("size of model: ",best_result_bohb_incremental[5])
print("config: ", best_result_bohb_incremental[6])
print("budget:", best_result_bohb_incremental[7])

# print("best_result_bohb_meta:")
# print("loss: ", best_result_bohb_meta[0])
# print("test accurancy: ", best_result_bohb_meta[2])
# print("trining time:", best_result_bohb_meta[3])
# print("record time:", best_result_bohb_meta[4])
# print("size of model: ",best_result_bohb_meta[5])
# print("config: ", best_result_bohb_meta[6])
# print("budget:", best_result_bohb_meta[7])

x_axis_bohb = results_bohb[:, 4]
y_axis_bohb = results_bohb[:, 2]

x_axis_bohb_incremental = results_bohb_incremental[:, 4]
y_axis_bohb_incremental = results_bohb_incremental[:, 2]

# x_axis_bohb_meta = results_bohb_meta[:, 4]
# y_axis_bohb_meta = results_bohb_meta[:, 2]

# x_axis_bo_emukit = results_bo_emukit[:, 3]
# y_axis_bo_emukit = results_bo_emukit[:, 1]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x_axis_bohb, y_axis_bohb, 'rx-', label='bohb_with_constraint')  # Plot some data on the axes.
ax.plot(x_axis_bohb_incremental, y_axis_bohb_incremental, 'go-', label='bohb_with_constraint_incremental')  # Plot some data on the axes.
# ax.plot(x_axis_bohb_meta, y_axis_bohb_meta, 'bv-', label='bohb_with_constraint_meta')
ax.plot(o_x_axis, o_y_axis, 'mo-', label='original bo(emukit)')  # Plot some data on the axes.
ax.plot(n_x_axis, n_y_axis, 'yo-', label='aggregation bo(emukit)')
#ax.plot(spearmint_result_xaxis, spearmint_result_yaxis, 'x-', label='spearmint with constraint %s samples' % format(spearmint_result_num))  # Plot more data on the axes...
# ax.plot(x_axis_bo_emukit, y_axis_bo_emukit, 'bx-', label='bo_emukit_with_constraint')
ax.set_xlabel('running time (s)')  # Add an x-label to the axes.
ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
ax.set_title("compare methods with constraint training time 1000s and model size 200MB")  # Add a title to the axes.
ax.legend()
plt.show()

