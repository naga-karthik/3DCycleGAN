# for plotting a boxplot of mean MI values for axial, sagittal and coronal slices
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# values = {'Mean Axial MI': [0.4356, 0.4763, 0.3984, 0.5431, 0.3643, 0.4540, 0.4306, 0.4102, 0.4717, 0.4342,
#                             0.4349, 0.3927, 0.4055, 0.4742, 0.3842],
#           'Mean Sagittal MI': [0.2692, 0.3194, 0.1832, 0.3821, 0.1962, 0.2656, 0.2486, 0.2160, 0.3059, 0.2556,
#                                0.2658, 0.2145, 0.2242, 0.3268, 0.2218],
#           'Mean Coronal MI': [0.2370, 0.3176, 0.2873, 0.3795, 0.2230, 0.2185, 0.2670, 0.2684, 0.2607, 0.2871,
#                               0.2999,  0.2351, 0.2930, 0.3078, 0.2518]
#           }
# df = pd.DataFrame(data=values)
values = [ ['Axial', 0.4356], ['Axial', 0.4763], ['Axial', 0.3984], ['Axial', 0.5431], ['Axial', 0.3643],
           ['Axial', 0.4540], ['Axial', 0.4306], ['Axial', 0.4102], ['Axial', 0.4717], ['Axial', 0.4342],
           ['Axial', 0.4349], ['Axial', 0.3927], ['Axial', 0.4055], ['Axial', 0.4742], ['Axial', 0.3842],

           ['Sagittal', 0.2692], ['Sagittal', 0.3194], ['Sagittal', 0.1832], ['Sagittal', 0.3821],
           ['Sagittal', 0.1962], ['Sagittal', 0.2656], ['Sagittal', 0.2486], ['Sagittal', 0.2160],
           ['Sagittal', 0.3059], ['Sagittal', 0.2556], ['Sagittal', 0.2658], ['Sagittal', 0.2145],
           ['Sagittal', 0.2242], ['Sagittal', 0.3268], ['Sagittal', 0.2218],

           ['Coronal', 0.2370], ['Coronal', 0.3176], ['Coronal', 0.2873], ['Coronal', 0.3795], ['Coronal', 0.2230],
           ['Coronal', 0.2185], ['Coronal', 0.2670], ['Coronal', 0.2684], ['Coronal', 0.2607], ['Coronal', 0.2871],
           ['Coronal', 0.2999], ['Coronal', 0.2351], ['Coronal', 0.2930], ['Coronal', 0.3078], ['Coronal', 0.2518]
          ]
df = pd.DataFrame(data=values, columns=['Slice Type', 'Mean MI Score'])
print(df.head())

sns.set_style(style='darkgrid')
sns.boxplot(x="Slice Type", y="Mean MI Score", data=df)
plt.show()


