import numpy as np

# # patient 12's data
# mean_distances = np.array((4.8675, 3.5612, 3.4475, 5.4267, 5.2508, 4.1865, 4.2957, 3.9883, 4.1433))
# mean_p12 = np.mean(mean_distances)      # 4.3519 mm
# std_p12 = np.std(mean_distances)        # 0.6565 mm

# level T2, 3 vertebrae
mean_distances_t2 = np.array((3.68, 5.42, 5.78))
mean_t2 = np.mean(mean_distances_t2)
std_t2 = np.std(mean_distances_t2)

# level T3, 3 vertebrae
mean_distances_t3 = np.array((2.23, 3.23, 5.48))
mean_t3 = np.mean(mean_distances_t3)
std_t3 = np.std(mean_distances_t3)

# level T4, 3 vertebrae
mean_distances_t4 = np.array((2.56, 2.84, 3.67))
mean_t4 = np.mean(mean_distances_t4)
std_t4 = np.std(mean_distances_t4)

# level T5, 3 vertebrae
mean_distances_t5 = np.array((3.02, 3.12, 3.61))
mean_t5 = np.mean(mean_distances_t5)
std_t5 = np.std(mean_distances_t5)

# level T6, 3 vertebrae
mean_distances_t6 = np.array((2.73, 2.76, 3.34))
mean_t6 = np.mean(mean_distances_t6)
std_t6 = np.std(mean_distances_t6)

# level T7, 3 vertebrae
mean_distances_t7 = np.array((2.77, 2.23, 5.47))
mean_t7 = np.mean(mean_distances_t7)
std_t7 = np.std(mean_distances_t7)

# level T8, 3 vertebrae
mean_distances_t8 = np.array((2.64, 2.01, 4.59))
mean_t8 = np.mean(mean_distances_t8)
std_t8 = np.std(mean_distances_t8)

# level T9, 3 vertebrae
mean_distances_t9 = np.array((3.07, 2.82, 3.31))
mean_t9 = np.mean(mean_distances_t9)
std_t9 = np.std(mean_distances_t9)

# level T10, 3 vertebrae
mean_distances_t10 = np.array((3.15, 2.34, 4.81))
mean_t10 = np.mean(mean_distances_t10)
std_t10 = np.std(mean_distances_t10)

# level T11, 1 vertebra
mean_distances_t11 = np.array((2.89))
mean_t11 = np.mean(mean_distances_t11)
std_t11 = np.std(mean_distances_t11)

# final
all_distances = np.array((mean_t2, mean_t3, mean_t4, mean_t5, mean_t6, mean_t7, mean_t8, mean_t9, mean_t10, mean_t11))
all_mean = np.mean(all_distances)
all_std = np.std(all_distances)