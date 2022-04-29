import numpy as np
import matplotlib.pyplot as plt

# results_bo_emukit = np.load('./running_results_bo_emukit_with_constraint_15s.npy', allow_pickle=True)
# results_bo_emukit = np.delete(results_bo_emukit, 0, 0)
# best_result_bo_emukit = results_bo_emukit[results_bo_emukit[:, 0].argsort()][0]

def universal_time_conversion(data, times, all_time):
    new_array = np.zeros(len(all_time))

    i_all = 0
    i_local = 0
    while i_all < len(all_time):
        new_array[i_all] = data[i_local]
        i_all += 1
        #if i_all < len(all_time):
        if i_local + 1 < len(times):
            if all_time[i_all] < times[i_local + 1]:
                pass
            else:
                i_local += 1
    return new_array

results_bohb_1 = np.load('./running_results_based_500_50.npy', allow_pickle=True)
results_bohb_1 = np.delete(results_bohb_1, 0, 0)
best_result_bohb_1 = results_bohb_1[-1, :]

results_bohb_2 = np.load('./running_results_based_500_50_1.npy', allow_pickle=True)
results_bohb_2 = np.delete(results_bohb_2, 0, 0)
best_result_bohb_2 = results_bohb_2[-1, :]

results_bohb_3 = np.load('./running_results_based_500_50_2.npy', allow_pickle=True)
results_bohb_3 = np.delete(results_bohb_3, 0, 0)
best_result_bohb_3 = results_bohb_3[-1, :]


results_bohb_incremental_1 = np.load('./running_results_inc_500_50.npy', allow_pickle=True)
results_bohb_incremental_1 = np.delete(results_bohb_incremental_1, 0, 0)
best_result_bohb_incremental_1 = results_bohb_incremental_1[-1, :]

results_bohb_incremental_2 = np.load('./running_results_inc_500_50_1.npy', allow_pickle=True)
results_bohb_incremental_2 = np.delete(results_bohb_incremental_2, 0, 0)
best_result_bohb_incremental_2 = results_bohb_incremental_2[-1, :]

results_bohb_incremental_3 = np.load('./running_results_inc_500_50_2.npy', allow_pickle=True)
results_bohb_incremental_3 = np.delete(results_bohb_incremental_3, 0, 0)
best_result_bohb_incremental_3 = results_bohb_incremental_3[-1, :]



# results_bohb_meta = np.load('./bohb_running_results_inc_and_meta_cnn_MNIST_5_0.5.npy', allow_pickle=True)
# results_bohb_meta = np.delete(results_bohb_meta, 0, 0)
# best_result_bohb_meta = results_bohb_meta[-1, :]
# print("best_result_bo_emukit:")
# print("loss: ", best_result_bo_emukit[0])
# print("test accurancy: ", best_result_bo_emukit[1])
# print("trining time:", best_result_bo_emukit[2])
# print("record time:", best_result_bo_emukit[3])
# print("config: ", best_result_bo_emukit[4])


o_results_1 = np.load('./running_results_bo_cnn6_cifar10_500_50.npy', allow_pickle=True)
o_re_1 = np.delete(o_results_1, 0, 0)

o_results_2 = np.load('./running_results_bo_cnn6_cifar10_500_50_1.npy', allow_pickle=True)
o_re_2 = np.delete(o_results_2, 0, 0)

o_results_3 = np.load('./running_results_bo_cnn6_cifar10_500_50_2.npy', allow_pickle=True)
o_re_3 = np.delete(o_results_3, 0, 0)

n_results_1 = np.load('./running_results_new_cnn6_cifar10_500_50.npy', allow_pickle=True)
n_re_1 = np.delete(n_results_1, 0, 0)

n_results_2 = np.load('./running_results_new_cnn6_cifar10_500_50_1.npy', allow_pickle=True)
n_re_2 = np.delete(n_results_2, 0, 0)

n_results_3 = np.load('./running_results_new_cnn6_cifar10_500_50_2.npy', allow_pickle=True)
n_re_3 = np.delete(n_results_3, 0, 0)


#print(results)
o_best_result = o_re_1[o_re_1[:, 0].argsort()][0]
print("original best result:")
print("loss: ", o_best_result[0])
print("test accurancy: ", o_best_result[1])
print("trining time:", o_best_result[2])
print("record time:", o_best_result[3])
print("config: ", o_best_result[4])

n_best_result = n_re_1[n_re_1[:, 0].argsort()][0]
print("best result:")
print("loss: ", n_best_result[0])
print("test accurancy: ", n_best_result[1])
print("trining time:", n_best_result[2])
print("record time:", n_best_result[3])
print("config: ", n_best_result[4])


print("best_result_bohb:")
print("loss: ", best_result_bohb_1[0])
print("test accurancy: ", best_result_bohb_1[2])
print("trining time:", best_result_bohb_1[3])
print("record time:", best_result_bohb_1[4])
print("size of model: ", best_result_bohb_1[5])
print("config: ", best_result_bohb_1[6])
print("budget:", best_result_bohb_1[7])

print("best_result_bohb_incremental:")
print("loss: ", best_result_bohb_incremental_1[0])
print("test accurancy: ", best_result_bohb_incremental_1[2])
print("trining time:", best_result_bohb_incremental_1[3])
print("record time:", best_result_bohb_incremental_1[4])
print("size of model: ",best_result_bohb_incremental_1[5])
print("config: ", best_result_bohb_incremental_1[6])
print("budget:", best_result_bohb_incremental_1[7])

# print("best_result_bohb_meta:")
# print("loss: ", best_result_bohb_meta[0])
# print("test accurancy: ", best_result_bohb_meta[2])
# print("trining time:", best_result_bohb_meta[3])
# print("record time:", best_result_bohb_meta[4])
# print("size of model: ",best_result_bohb_meta[5])
# print("config: ", best_result_bohb_meta[6])
# print("budget:", best_result_bohb_meta[7])

o_time_1 = o_re_1[:, 3]
o_test_acc_1 = o_re_1[:, 1]

o_time_2 = o_re_2[:, 3]
o_test_acc_2 = o_re_2[:, 1]

o_time_3 = o_re_3[:, 3]
o_test_acc_3 = o_re_3[:, 1]

o_time = np.sort(list(set(o_time_1).union(o_time_2, o_time_3)))

o_test_acc_1 = universal_time_conversion(o_test_acc_1, o_time_1, o_time)
o_test_acc_2 = universal_time_conversion(o_test_acc_2, o_time_2, o_time)
o_test_acc_3 = universal_time_conversion(o_test_acc_3, o_time_3, o_time)

o_test_acc = (o_test_acc_1 + o_test_acc_2 + o_test_acc_3) / 3


n_time_1 = n_re_1[:, 3]
n_test_acc_1 = n_re_1[:, 1]

n_time_2 = n_re_2[:, 3]
n_test_acc_2 = n_re_2[:, 1]

n_time_3 = n_re_3[:, 3]
n_test_acc_3 = n_re_3[:, 1]

n_time = np.sort(list(set(n_time_1).union(n_time_2, n_time_3)))

n_test_acc_1 = universal_time_conversion(n_test_acc_1, n_time_1, n_time)
n_test_acc_2 = universal_time_conversion(n_test_acc_2, n_time_2, n_time)
n_test_acc_3 = universal_time_conversion(n_test_acc_3, n_time_3, n_time)

n_test_acc = (n_test_acc_1 + n_test_acc_2 + n_test_acc_3) / 3


x_bohb_time_1 = results_bohb_1[:, 4]
y_bohb_test_acc_1 = results_bohb_1[:, 2]

x_bohb_time_2 = results_bohb_2[:, 4]
y_bohb_test_acc_2 = results_bohb_2[:, 2]


x_bohb_time_3 = results_bohb_3[:, 4]
y_bohb_test_acc_3 = results_bohb_3[:, 2]

x_bohb_time = np.sort(list(set(x_bohb_time_1).union(x_bohb_time_2, x_bohb_time_3)))

y_bohb_test_acc_1 = universal_time_conversion(y_bohb_test_acc_1, x_bohb_time_1, x_bohb_time)
y_bohb_test_acc_2 = universal_time_conversion(y_bohb_test_acc_2, x_bohb_time_2, x_bohb_time)
y_bohb_test_acc_3 = universal_time_conversion(y_bohb_test_acc_3, x_bohb_time_3, x_bohb_time)

y_bohb_test_acc = (y_bohb_test_acc_1 + y_bohb_test_acc_2 + y_bohb_test_acc_3) / 3



x_inc_time_1 = results_bohb_incremental_1[:, 4]
y_inc_test_acc_1 = results_bohb_incremental_1[:, 2]

x_inc_time_2 = results_bohb_incremental_2[:, 4]
y_inc_test_acc_2 = results_bohb_incremental_2[:, 2]

x_inc_time_3 = results_bohb_incremental_3[:, 4]
y_inc_test_acc_3 = results_bohb_incremental_3[:, 2]

x_inc_time = np.sort(list(set(x_inc_time_1).union(x_inc_time_2, x_inc_time_3)))

y_inc_test_acc_1 = universal_time_conversion(y_inc_test_acc_1, x_inc_time_1, x_inc_time)
y_inc_test_acc_2 = universal_time_conversion(y_inc_test_acc_2, x_inc_time_2, x_inc_time)
y_inc_test_acc_3 = universal_time_conversion(y_inc_test_acc_3, x_inc_time_3, x_inc_time)

y_inc_test_acc = (y_inc_test_acc_1 + y_inc_test_acc_2 + y_inc_test_acc_3) / 3


# x_axis_bohb_meta = results_bohb_meta[:, 4]
# y_axis_bohb_meta = results_bohb_meta[:, 2]

# x_axis_bo_emukit = results_bo_emukit[:, 3]
# y_axis_bo_emukit = results_bo_emukit[:, 1]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x_bohb_time, y_bohb_test_acc, 'rx-', label='bohb_with_constraint')  # Plot some data on the axes.
ax.plot(x_inc_time, y_inc_test_acc, 'go-', label='bohb_with_constraint_incremental')  # Plot some data on the axes.
# ax.plot(x_axis_bohb_meta, y_axis_bohb_meta, 'bv-', label='bohb_with_constraint_meta')
ax.plot(o_time, o_test_acc, 'mo-', label='original bo(emukit)')  # Plot some data on the axes.
ax.plot(n_time, n_test_acc, 'yo-', label='aggregation bo(emukit)')
#ax.plot(spearmint_result_xaxis, spearmint_result_yaxis, 'x-', label='spearmint with constraint %s samples' % format(spearmint_result_num))  # Plot more data on the axes...
# ax.plot(x_axis_bo_emukit, y_axis_bo_emukit, 'bx-', label='bo_emukit_with_constraint')
ax.set_xlabel('running time (s)')  # Add an x-label to the axes.
ax.set_ylabel('avg test accuracy')  # Add a y-label to the axes.
ax.set_title("compare methods with constraint training time 500s and model size 50MB")  # Add a title to the axes.
ax.legend()
plt.show()

