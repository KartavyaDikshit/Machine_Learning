import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('chat_dataset.csv', delimiter='\t')

# Check the column names in your dataset
print(data.columns)

# Based on the column names, select the appropriate columns
X = data['message']  # Adjust this to the actual column name for messages in your dataset
y = data['sentiment']

# Create a bar chart for sentiment distribution
sentiment_counts = y.value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.show()
