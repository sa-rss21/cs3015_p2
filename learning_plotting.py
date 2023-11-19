import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('grid_search_results_NR.csv')

# Step 2: Take the absolute values of mean_test_score
df['mean_test_score'] = df['mean_test_score'].abs()

# Step 3: Sort the DataFrame by mean_test_score in descending order
df = df.sort_values(by='mean_test_score', ascending=False)

# Step 4: Create a scatter plot of the sorted absolute mean test scores
plt.figure(figsize=(12, 6))
mean_test_scores = df['mean_test_score']
plt.scatter(range(len(mean_test_scores)), mean_test_scores)
plt.xlabel('Training Index')
plt.ylabel('VBS-SBS Gap Score')
plt.title('Sorted Cross-validated Gap Scores')
plt.show()