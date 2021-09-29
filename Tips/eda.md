# EDA Visualization Tips

Based on Pandas.DataFrame.

### Single column distribution

+ Numerical

```python
# Use matplotlib
fig, ax = plt.subplots(2,1, figsize=(9,12))
fig.suptitle('title')

ax[0].hist(label, bins=100)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_tile('title 1')

ax[0].axvline(label.mean(), color='r', linestyle='dashed', linewidth=2)

ax[1].boxplot(label, vert=False)
plt.show()
```

```python
# Use pandas 
df[col].hist(bins=100)
'''
Show the density
'''
df[col].plot.density()
```

+ Categorical

```python
fig = plt.figure(figsize=(9,6))
ax = fig.gca()

counts = df[col].value_counts() # sorted descendinly by default
counts = counts.sort_index() 
# counts = counts.sort_values()

df[col].plot.bar(ax=ax, color='b')
# df.plot.bar(x='Name', y='Target')
# df.plot.bar(x='Name', y=['Target 1', 'Target 2'])

ax.set_balabala
plt.show()
```

### Two columns dependency

+ Categorical vs Numerical

```python
fig = plt.figure(figszie=(9,6))
ax = fig.gca()

df.boxplot(column='numerical', by='categorical', ax=ax)
ax.balabala

plt.show()
```

+ Numerical vs Numerical

```python
fig = plt.figure(figsize=(9,6))
ax = fig.gca()

correlation = df['num1'].corr(df['num2'])

plt.scatter(df['num1'], df['num2'])
ax.set_title(f'num 1 vs num2 - correlation: {str(correlation)}')

plt.show()
```

+ Groupby trick

```python
df_group = df.groupby('Name')
target = df_group['Feature'].mean()
# target = df_group['Feature'].sum()

target = target.sort_values('Feature')

target.plot.bar()
```

### Data Frame Correlation

```python
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

sns.heatmap(corr, cmap='coolwarm')
```

### Pairplot

```python
sns.pairplot(df, hue='labels', kind='', diag_kind='')
```

```python
sns.jointplot(df, x='column x', y='column y', hue='labels', kind='')
```



