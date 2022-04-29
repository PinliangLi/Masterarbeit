import pickle
import logging
import sys
import matplotlib.pyplot as plt
#f = open('./bohb_with_constraints/result.txt', 'w')
import numpy as np

# res = pickle.load(open("./bohb_with_constraints/results.pkl", "rb"))
# id2config = res.get_id2config_mapping()
# incumbent, infos = res.get_incumbent_id()
#
# results = []
#
# for i, info in enumerate(infos):
#     if info[0] != np.inf:
#         #xaxis = np.append(xaxis, info[2]['record_time'])
#         #yaxis = np.append(yaxis, info[2]['validation accuracy'])
#         results.append([info[2]['record_time'], info[2]['test accuracy']])
#
# results = np.array(results)
# xaxis = results[:, 0]
# yaxis = results[:, 1]
# np.save('time_stamp.npy', results)
#
# plt.plot(xaxis, yaxis, 'o-')
# plt.show()
# #plt.savefig('foo.png')

# print('Best found configuration:', id2config[incumbent])
# print('result:', min(infos)[2])
# print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
# print('A total of %i runs where executed.' % len(res.get_all_runs()))
# print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/9))

re = np.load('./running_results_cnn_MNIST_1.npy', allow_pickle=True)
print(re)
re = np.delete(re, 0, 0)
best_result = re[-1, :]
print("best result:")
print("loss: ", best_result[0])
print("test accurancy: ", best_result[1])
print("training time:", best_result[2])
print("record time:", best_result[3])
print("config: ", best_result[5])
print("budget:", best_result[6])



x_axis = re[:, 3]
y_axis = re[:, 1]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x_axis, y_axis, 'o-', label='bohb')  # Plot some data on the axes.
#ax.plot(spearmint_result_xaxis, spearmint_result_yaxis, 'x-', label='spearmint with constraint %s samples' % format(spearmint_result_num))  # Plot more data on the axes...
ax.set_xlabel('running time (s)')  # Add an x-label to the axes.
ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
ax.set_title("compare methods")  # Add a title to the axes.
ax.legend()
plt.show()