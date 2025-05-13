import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your file
df = pd.read_csv('./macroF1.csv', index_col=0)  # Assuming first column is method names
df.index = df.index.tolist()
# Melt the dataframe to long format (good for grouped barplot)
df_melted = df.reset_index().melt(id_vars='index', var_name='Dataset', value_name='macroF1')
df_melted = df_melted.rename(columns={'index': 'Method'})

# Define colors: red for bioPointNet, grey for others

# Plot
fig = plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_melted,
    x='Dataset',
    y='macroF1',
    hue='Method',
    dodge=True
)

plt.title('Benchmarking of Methods across Datasets')
plt.ylabel('macroF1')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Method')
plt.tight_layout()
sns.despine()
fig.savefig('macroF1.pdf')