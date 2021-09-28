import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sns.set()

simulation_data = pd.read_csv('simulation_data_new_all.csv', sep=';').dropna().iloc[::17, :]
simulation_data['actuation_difference'] = simulation_data['actuation'] - simulation_data['previous_actuation']

plt.figure()
plot_sim_data = pd.read_csv('simulation_data_new_all.csv', sep=';')
sns.lineplot(x='datetime', y='temperature', hue='room_name', ci=None, data=plot_sim_data[plot_sim_data['timeline'] < 14400])
del plot_sim_data

plt.figure()
sns.kdeplot(x='actuation', hue='room_name', data=simulation_data)
plt.figure()
sns.relplot(x='previous_actuation', y='actuation', hue='room_name', data=simulation_data)
plt.figure()
sns.kdeplot(x='previous_actuation', y='actuation', data=simulation_data)
print(simulation_data[['previous_actuation', 'actuation']].corr())
print(simulation_data['actuation_difference'].std())
#sns.pairplot(simulation_data.iloc[::7, :], vars=['actuation', 'previous_actuation'] , hue='room_name', plot_kws={'linewidth': 0.1})
plt.figure()
sns.kdeplot(x='actuation_difference', data=simulation_data)
cuts = pd.qcut(simulation_data['actuation_difference'], 11).value_counts(normalize=True, ascending=True, sort=False)

plt.figure()
sns.histplot(x='actuation_difference', bins=3, data=simulation_data)
print(cuts)
if __name__ == '__main__':
    plt.show()