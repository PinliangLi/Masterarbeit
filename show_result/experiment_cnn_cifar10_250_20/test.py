import numpy as np
import matplotlib.pyplot as plt

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

_exp_num = 5
exp_num = np.arange(1, _exp_num + 1)

new_method_results = []
o_results = []

results_bohbs = []
results_bohb_incs = []
results_bohb_inc_metas = []
results_before_trains = []
results_before_train_abss = []
results_use_distances = []

print(exp_num)

for i in exp_num:

    new_method_file = './running_results_new_cnn6_cifar_250_20_time_version_' + str(i) + '.npy'
    new_method_results.append(np.load(new_method_file, allow_pickle=True))
    new_method_results[i - 1] = np.delete(new_method_results[i - 1], 0, 0)

    o_results_file = './running_results_bo_cnn6_cifar_250_20_time_version_' + str(i) + '.npy'
    o_results.append(np.load(o_results_file, allow_pickle=True))
    o_results[i - 1] = np.delete(o_results[i - 1], 0, 0)

    results_bohb_file = './running_results_based_' + str(i) + '.npy'
    results_bohbs.append(np.load(results_bohb_file, allow_pickle=True))
    results_bohbs[i - 1] = np.delete(results_bohbs[i - 1], 0, 0)

    results_bohb_inc_file = './running_results_inc_' + str(i) + '.npy'
    results_bohb_incs.append(np.load(results_bohb_inc_file, allow_pickle=True))
    results_bohb_incs[i - 1] = np.delete(results_bohb_incs[i - 1], 0, 0)

    results_bohb_inc_meta_file = './running_results_inc_meta_' + str(i) + '.npy'
    results_bohb_inc_metas.append(np.load(results_bohb_inc_meta_file, allow_pickle=True))
    results_bohb_inc_metas[i - 1] = np.delete(results_bohb_inc_metas[i - 1], 0, 0)

    results_before_trains_file = './running_results_inc_before_' + str(i) + '.npy'
    results_before_trains.append(np.load(results_before_trains_file, allow_pickle=True))
    results_before_trains[i - 1] = np.delete(results_before_trains[i - 1], 0, 0)

    results_before_trains_abs_file = './running_results_inc_before_abs_' + str(i) + '.npy'
    results_before_train_abss.append(np.load(results_before_trains_abs_file, allow_pickle=True))
    results_before_train_abss[i - 1] = np.delete(results_before_train_abss[i - 1], 0, 0)

    results_use_distances_file = './running_results_inc_dist_' + str(i) + '.npy'
    results_use_distances.append(np.load(results_use_distances_file, allow_pickle=True))
    results_use_distances[i - 1] = np.delete(results_use_distances[i - 1], 0, 0)

n_times = []
o_times = []

x_bohb_times = []
x_inc_times = []
x_inc_meta_times = []
x_before_times = []
x_before_abs_times = []
x_use_distance_times = []

for i in exp_num:

    n_times.append(new_method_results[i - 1][:, 3])
    o_times.append(o_results[i - 1][:, 3])

    x_bohb_times.append(results_bohbs[i - 1][:, 4])
    x_inc_times.append(results_bohb_incs[i - 1][:, 4])
    x_inc_meta_times.append(results_bohb_inc_metas[i - 1][:, 4])
    x_before_times.append(results_before_trains[i - 1][:, 4])
    x_before_abs_times.append(results_before_train_abss[i - 1][:, 4])
    x_use_distance_times.append(results_use_distances[i - 1][:, 4])

    if i == 1:
        n_time = set(n_times[i - 1])
        o_time = set(o_times[i - 1])

        x_bohb_time = set(x_bohb_times[i - 1])
        x_inc_time = set(x_inc_times[i - 1])
        x_inc_meta_time = set(x_inc_meta_times[i - 1])
        x_before_time = set(x_before_times[i - 1])
        x_before_abs_time = set(x_before_abs_times[i - 1])
        x_use_distance_time = set(x_use_distance_times[i - 1])

    else:
        n_time = n_time.union(n_times[i - 1])
        o_time = o_time.union(o_times[i - 1])

        x_bohb_time = x_bohb_time.union(x_bohb_times[i - 1])
        x_inc_time = x_inc_time.union(x_inc_times[i - 1])
        x_inc_meta_time = x_inc_meta_time.union(x_inc_meta_times[i - 1])
        x_before_time = x_before_time.union(x_before_times[i - 1])
        x_before_abs_time = x_before_abs_time.union(x_before_abs_times[i - 1])
        x_use_distance_time = x_use_distance_time.union(x_use_distance_times[i - 1])

n_time = np.sort(list(n_time))
o_time = np.sort(list(o_time))

x_bohb_time = np.sort(list(x_bohb_time))
x_inc_time = np.sort(list(x_inc_time))
x_inc_meta_time = np.sort(list(x_inc_meta_time))
x_before_time = np.sort(list(x_before_time))
x_before_abs_time = np.sort(list(x_before_abs_time))
x_use_distance_time = np.sort(list(x_use_distance_time))

for i in exp_num:
    if i == 1:
        n_test_acc = universal_time_conversion(new_method_results[i - 1][:, 1], n_times[i - 1], n_time)
        o_test_acc = universal_time_conversion(o_results[i - 1][:, 1], o_times[i - 1], o_time)

        y_bohb_test_acc = universal_time_conversion(results_bohbs[i - 1][:, 2], x_bohb_times[i - 1], x_bohb_time)
        y_inc_test_acc = universal_time_conversion(results_bohb_incs[i - 1][:, 2], x_inc_times[i - 1], x_inc_time)
        y_inc_meta_test_acc = universal_time_conversion(results_bohb_inc_metas[i - 1][:, 2], x_inc_meta_times[i - 1], x_inc_meta_time)
        y_before_test_acc = universal_time_conversion(results_before_trains[i - 1][:, 2], x_before_times[i - 1],
                                                        x_before_time)
        y_before_abs_test_acc = universal_time_conversion(results_before_train_abss[i - 1][:, 2], x_before_abs_times[i - 1],
                                                      x_before_abs_time)
        y_use_distance_test_acc = universal_time_conversion(results_use_distances[i - 1][:, 2],
                                                          x_use_distance_times[i - 1],
                                                          x_use_distance_time)



    else:
        n_test_acc = n_test_acc + universal_time_conversion(new_method_results[i - 1][:, 1], n_times[i - 1], n_time)
        o_test_acc = o_test_acc + universal_time_conversion(o_results[i - 1][:, 1], o_times[i - 1], o_time)

        y_bohb_test_acc = y_bohb_test_acc + universal_time_conversion(results_bohbs[i - 1][:, 2], x_bohb_times[i - 1], x_bohb_time)
        y_inc_test_acc = y_inc_test_acc + universal_time_conversion(results_bohb_incs[i - 1][:, 2], x_inc_times[i - 1], x_inc_time)
        y_inc_meta_test_acc = y_inc_meta_test_acc + universal_time_conversion(results_bohb_inc_metas[i - 1][:, 2], x_inc_meta_times[i - 1],
                                                        x_inc_meta_time)
        y_before_test_acc = y_before_test_acc + universal_time_conversion(results_before_trains[i - 1][:, 2], x_before_times[i - 1],
                                                      x_before_time)
        y_before_abs_test_acc = y_before_abs_test_acc + universal_time_conversion(results_before_train_abss[i - 1][:, 2], x_before_abs_times[i - 1],
                                                      x_before_abs_time)
        y_use_distance_test_acc = y_use_distance_test_acc + universal_time_conversion(results_use_distances[i - 1][:, 2],
                                                            x_use_distance_times[i - 1],
                                                            x_use_distance_time)

n_test_acc = n_test_acc / _exp_num
o_test_acc = o_test_acc / _exp_num

y_bohb_test_acc = y_bohb_test_acc / _exp_num
y_inc_test_acc = y_inc_test_acc / _exp_num
y_inc_meta_test_acc = y_inc_meta_test_acc / _exp_num
y_before_test_acc = y_before_test_acc / _exp_num
y_before_abs_test_acc = y_before_abs_test_acc / _exp_num
y_use_distance_test_acc = y_use_distance_test_acc / _exp_num

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

fig.suptitle('compare methods with constraint training time 250s and model size 20MB')

axs[0, 0].plot(o_time, o_test_acc, 'mo-', label='original bo(emukit)')  # Plot some data on the axes.
axs[0, 0].plot(n_time, n_test_acc, 'yo-', label='aggregation bo(emukit)')
axs[0, 0].plot(x_bohb_time, y_bohb_test_acc, 'bx-', label='bohb_with_constraint')  # Plot some data on the axes.
axs[0, 0].legend(loc='lower right')


axs[0, 1].plot(x_bohb_time, y_bohb_test_acc, 'bx-', label='bohb_with_constraint')  # Plot some data on the axes.
axs[0, 1].plot(x_inc_time, y_inc_test_acc, 'gx-', label='bohb_with_constraint_incremental')  # Plot some data on the axes.
axs[0, 1].legend(loc='lower right')

axs[0, 2].plot(x_bohb_time, y_bohb_test_acc, 'bx-', label='bohb_with_constraint')
axs[0, 2].plot(x_inc_meta_time, y_inc_meta_test_acc, 'rx-', label='bohb_with_constraint_meta')
axs[0, 2].legend(loc='lower right')

axs[1, 0].plot(x_bohb_time, y_bohb_test_acc, 'bx-', label='bohb_with_constraint')
axs[1, 0].plot(x_before_time, y_before_test_acc, 'cv-', label='early stop')
axs[1, 0].legend(loc='lower right')

axs[1, 1].plot(x_bohb_time, y_bohb_test_acc, 'bx-', label='bohb_with_constraint')
axs[1, 1].plot(x_use_distance_time, y_use_distance_test_acc, 'kv-', label='use distance instead of infinity')
axs[1, 1].legend(loc='lower right')

axs[1, 2].plot(x_bohb_time, y_bohb_test_acc, 'bx-', label='bohb_with_constraint')
axs[1, 2].plot(x_before_abs_time, y_before_abs_test_acc, 'yv-', label='early stop and set lower starting budget')
axs[1, 2].legend(loc='lower right')

for ax in axs.flat:
    ax.set(xlabel='runing time (seconds)', ylabel='test accurancy')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()

# ax1.plot(x_inc_time, y_inc_test_acc, 'gx-', label='bohb_with_constraint_incremental')
# ax1.plot(x_inc_meta_time, y_inc_meta_test_acc, 'rx-', label='bohb_with_constraint_meta')
# ax1.legend(loc='lower right')
#
# ax.plot(x_before_time, y_before_test_acc, 'cv-', label='early stop by checking size before training and checking training time during training')  # Plot some data on the axes.
# ax.plot(x_before_abs_time, y_before_abs_test_acc, 'yv-', label='early stop and use absolute data budget')  # Plot some data on the axes.
# ax.plot(x_use_distance_time, y_use_distance_test_acc, 'kv-', label='use distance instead of infinity')  # Plot some data on the axes.
# #
# ax.plot(o_time, o_test_acc, 'mo-', label='original bo(emukit)')  # Plot some data on the axes.
# ax.plot(n_time, n_test_acc, 'yo-', label='aggregation bo(emukit)')
# #ax.plot(spearmint_result_xaxis, spearmint_result_yaxis, 'x-', label='spearmint with constraint %s samples' % format(spearmint_result_num))  # Plot more data on the axes...
# # ax.plot(x_axis_bo_emukit, y_axis_bo_emukit, 'bx-', label='bo_emukit_with_constraint')
# ax.set_xlabel('running time (s)')  # Add an x-label to the axes.
# ax.set_ylabel('avg test accuracy')  # Add a y-label to the axes.
# ax0.set_title("compare methods with constraint training time 250s and model size 20MB")  # Add a title to the axes.

print(len(new_method_results))