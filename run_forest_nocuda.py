import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Ensure the model runs on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your data
forrest_df = pd.read_csv('Movies_Parties_cat_BIG.csv')  # Uncomment and modify this line to load your data

# Split the ProductionCompanies column into a list of companies for each row
all_comp_list = forrest_df['ProductionCompanies'].str.split(', ')
comp_count = all_comp_list.apply(len)
max_comps = comp_count.max()

# Flatten the list of lists into a single list of all companies
all_comps = [company for sublist in all_comp_list for company in sublist]
unique_companies = set(all_comps)

# Find the row with the maximum number of companies
max_comp_row = forrest_df.loc[comp_count.idxmax()].ProductionCompanies

# Split the ProductionCompanies column into multiple columns
split_df = forrest_df['ProductionCompanies'].str.split(', ', expand=True)

# Ensure there are 9 columns, filling with None if there are fewer companies
for i in range(9):
    if i >= split_df.shape[1]:
        split_df[i] = None

# Rename the new columns
split_df.columns = [f'ProductionCompany{i+1}' for i in range(9)]

# Concatenate the new columns with the original DataFrame
df_expand = pd.concat([forrest_df, split_df], axis=1)

# Drop the original ProductionCompanies column if desired
df_expand = df_expand.drop(columns=['ProductionCompanies'])

# Separate features and target (profit)
target = 'profit_ratio'
features_df = df_expand.drop(columns=[target, 'tconst', 'averageRating', 'primaryTitle'])
target_df = df_expand[target]

# One-hot encode the categorical features (actor names, genres, production companies...)
for column in features_df.select_dtypes(include=['object']).columns:
    features_df = pd.concat([features_df, pd.get_dummies(features_df[column], dummy_na=False)], axis=1)
    features_df.drop(column, axis=1, inplace=True)

# Ensure the target and features have the same rows
assert features_df.index.size == target_df.index.size, "mismatched rows"

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)  # Reshape target values
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)  # Reshape target values

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_dim=x_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    start_time = time.time()
    predictions = model(x_test_tensor)
    end_time = time.time()
    predictions = predictions.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    mse = mean_squared_error(y_test_np, predictions)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Time taken: {end_time - start_time} seconds")

# Define the permutation importances function
def permutation_importances(model, model_input, model_output, metric, n_repeats=2):
    baseline = metric(model_output.cpu().numpy(), model(model_input).cpu().detach().numpy())
    importances = np.zeros(model_input.shape[1])
    
    for column in tqdm(range(model_input.shape[1]), desc="Calculating importances"):
        score_array = np.zeros(n_repeats)
        
        for n in range(n_repeats):
            x_permuted = model_input.clone().detach()
            x_permuted[:, column] = torch.tensor(np.random.permutation(x_permuted[:, column].cpu().numpy())).to(device)
            score = metric(model_output.cpu().numpy(), model(x_permuted).cpu().detach().numpy())
            score_array[n] = baseline - score
        
        importances[column] = np.mean(score_array)
    
    return importances

# Run the permutation importances function
importance = permutation_importances(model, x_test_tensor, y_test_tensor, mean_squared_error)
feature_labels = features_df.columns

# Create a dataframe of the feature importances
importance_df = pd.DataFrame({
    'feature': feature_labels,
    'importance': importance
}).sort_values(by='importance', ascending=False)

# Display the top 10 most important features
print(importance_df.head(10))

# Save importance to file
importance_df.to_csv('Permu_importance_df.csv', index=False)

# Add the importance scores to the corresponding employee records
for feature in importance_df['feature']:
    df_expand[feature + '_importance'] = importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0]

# Save the merged dataframe with importance scores
df_expand.to_csv('employee_with_importance.csv', index=False)

# Aggregate the importance scores by category
categories = ['actor', 'producer', 'director', 'writer', 'ProductionCompany']
category_importance = {category: 0 for category in categories}

for feature in importance_df['feature']:
    for category in categories:
        if category in feature:
            category_importance[category] += importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0]

# Print the aggregated importance scores by category
print("Aggregated Importance Scores by Category:")
#save out put to file
for category, importance in category_importance.items():
    print(f"{category}: {importance}")
    with open('Aggregated_importanceNN.txt', 'a') as f:
        f.write(f"{category}: {importance}\n")
for category, importance in category_importance.items():
    print(f"{category}: {importance}")
