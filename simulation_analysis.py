import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

simulation_data = pd.read_csv('simulation_data_train.csv', sep=';').dropna().iloc[::17, :]
simulation_data['actuation_difference'] = simulation_data['actuation'] - simulation_data['previous_actuation']

plt.figure()
plot_sim_data = pd.read_csv('simulation_data_train.csv', sep=';')
sns.lineplot(x='datetime', y='room_temperature', hue='room_name', ci=None, data=plot_sim_data.iloc[:14400, :])
del plot_sim_data

plt.figure()
sns.kdeplot(x='actuation', hue='room_name', data=simulation_data)
plt.figure()
sns.relplot(x='previous_actuation', y='actuation', hue='room_name', data=simulation_data)
plt.figure()
sns.kdeplot(x='previous_actuation', y='actuation', data=simulation_data)
print(simulation_data[['previous_actuation', 'actuation']].corr())
#sns.pairplot(simulation_data.iloc[::7, :], vars=['actuation', 'previous_actuation'] , hue='room_name', plot_kws={'linewidth': 0.1})
plt.figure()
sns.kdeplot(x='actuation_difference', data=simulation_data)
cuts = pd.cut(simulation_data['actuation_difference'], 3).value_counts(normalize=True, ascending=True, sort=False)
plt.plot([cuts.index[0].left]*2, [0, 0.01], 'k')
plt.plot([cuts.index[0].right]*2, [0, 0.01], 'k')
plt.plot([cuts.index[2].left]*2, [0, 0.01], 'k')
plt.plot([cuts.index[2].right]*2, [0, 0.01], 'k')

plt.figure()
sns.histplot(x='actuation_difference', bins=3, data=simulation_data)
print(cuts)
if __name__ == '__main__':
    plt.show()