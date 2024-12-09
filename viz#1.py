import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('ObesityDataSet_modified.csv')

faf_counts = data.groupby(['NObeyesdad', 'FAF']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
faf_counts.plot(kind='bar', stacked=False, ax=ax)
ax.set_title('Physical Activity (FAF) by Obesity Levels')
ax.set_xlabel('Obesity Levels (NObeyesdad)')
ax.set_ylabel('Number of People')
ax.legend(['No Activity', 'Low Activity', 'Moderate Activity', 'High Activity'])
ax.set_xticklabels(faf_counts.index, rotation=45)
plt.tight_layout()
plt.show()


scc_counts = data.groupby(['NObeyesdad', 'SCC']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
scc_counts.plot(kind='bar', stacked=False, ax=ax)
ax.set_title('Calorie Tracking (SCC) by Obesity Levels')
ax.set_xlabel('Obesity Levels (NObeyesdad)')
ax.set_ylabel('Number of People')
ax.legend(['Not Tracking (SCC=0)', 'Tracking (SCC=1)'])
ax.set_xticklabels(scc_counts.index, rotation=45)
plt.tight_layout()
plt.show()


combined_data = data.groupby('NObeyesdad')[['FAF', 'SCC']].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
bar_positions = range(len(combined_data))
ax.bar([pos - bar_width/2 for pos in bar_positions], combined_data['FAF'], width=bar_width, label='Physical Activity (FAF)')
ax.bar([pos + bar_width/2 for pos in bar_positions], combined_data['SCC'], width=bar_width, label='Calorie Tracking (SCC)')
ax.set_xticks(bar_positions)
ax.set_xticklabels(combined_data['NObeyesdad'], rotation=45)
ax.set_xlabel('Obesity Levels (NObeyesdad)')
ax.set_ylabel('Mean Values')
ax.set_title('Physical Activity and Calorie Tracking Across Obesity Levels')
ax.legend()
plt.tight_layout()
plt.show()

