import numpy as np

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


run1_times = [0, 1, 3, 4, 6]
run1_test_accuracy = [0.0, 0.1, 0.3, 0.6, 0.67]

run2_times = [0, 1.5, 2, 4, 5, 6]
run2_test_accuracy = [0.0, 0.05, 0.4, 0.5, 0.7, 0.71]

test_all = np.sort(list(set(run1_times).union(run2_times)))

r1 = universal_time_conversion(run2_test_accuracy, run2_times, test_all)
r2 = universal_time_conversion(run1_test_accuracy, run1_times, test_all)

print(r1)
print(r2)

average_time = (r1 + r2) / 2.0
print('Time: ' + str(test_all))
print('Average Test Accuracy: ' + str(average_time))
