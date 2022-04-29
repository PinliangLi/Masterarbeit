import numpy as np
import matplotlib.pyplot as plt

o_results = np.load('./running_results_origin.npy', allow_pickle=True)
o_re = np.delete(o_results, 0, 0)
n_results = np.load('./running_results_new.npy', allow_pickle=True)
n_re = np.delete(n_results, 0, 0)

results_bohb = np.load('./running_results_based.npy', allow_pickle=True)
results_bohb = np.delete(results_bohb, 0, 0)

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

x_axis_bohb = results_bohb[:, 4]
y_axis_bohb = results_bohb[:, 2]


fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(o_x_axis, o_y_axis, 'ro-', label='original bo(emukit)')  # Plot some data on the axes.
ax.plot(n_x_axis, n_y_axis, 'bo-', label='new method bo(emukit)')
ax.plot(x_axis_bohb, y_axis_bohb, 'gx-', label='bohb_with_constraint')
#ax.plot(spearmint_result_xaxis, spearmint_result_yaxis, 'x-', label='spearmint with constraint %s samples' % format(spearmint_result_num))  # Plot more data on the axes...
ax.set_xlabel('running time (s)')  # Add an x-label to the axes.
ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
ax.set_title("compare methods in 100s and 10MB on RandomForest")  # Add a title to the axes.
ax.legend()
plt.show()