from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = pd.read_csv(r'E:\document\MIA\result.csv')  # Adjust path as needed

# Select features
similarity_distance_features = ['vitb32_75_similarity', 'vitb32_75_distance']
given_photo_feature = ['vitb32_75_given_photo']

# Determine which columns to use for training and testing
X_similarity_distance = data[similarity_distance_features]
X_given_photo = data[given_photo_feature]

y = data['inference'].apply(lambda x: 1 if x == 'out' else -1)  # Assuming 'inference' column contains labels

# Data standardization
scaler = StandardScaler()
X_similarity_distance_scaled = scaler.fit_transform(X_similarity_distance)
X_given_photo_scaled = scaler.fit_transform(X_given_photo)

# Split data into train and test sets for other four methods
X_train_sim_dist, X_test_sim_dist, y_train, y_test = train_test_split(X_similarity_distance_scaled, y, test_size=0.2, random_state=42)

# Train and predict using other four methods
iso_forest = IsolationForest(contamination=0.1)
lof = LocalOutlierFactor(novelty=True, contamination=0.1)
oc_svm = OneClassSVM(gamma='auto')

iso_forest.fit(X_train_sim_dist)
lof.fit(X_train_sim_dist)
oc_svm.fit(X_train_sim_dist)

y_pred_iso = iso_forest.predict(X_test_sim_dist)
y_pred_lof = lof.predict(X_test_sim_dist)
y_pred_svm = oc_svm.predict(X_test_sim_dist)


class Autoencoder(nn.Module):
    def __init__(self, input_size=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


autoencoder = Autoencoder(input_size=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)


X_tensor_train = torch.FloatTensor(X_train_sim_dist)
X_tensor_test = torch.FloatTensor(X_test_sim_dist)

train_dataset = TensorDataset(X_tensor_train, X_tensor_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)


epochs = 1000
for epoch in range(epochs):
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()


with torch.no_grad():
    reconstructed = autoencoder(X_tensor_test)
    mse_loss = nn.functional.mse_loss(reconstructed, X_tensor_test, reduction='none').mean(dim=1)
    mse_loss_np = mse_loss.numpy()
    threshold = np.quantile(mse_loss_np, 0.75)
    y_pred_ae = np.where(mse_loss_np > threshold, -1, 1)


# Train and predict using KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_given_photo_scaled)
centers = kmeans.cluster_centers_

# Check distance between cluster centers
distance_centers = np.linalg.norm(centers[0] - centers[1])

if distance_centers < 3:
    print("Clustering into two classes failed, this method is ineffective.")
    # Remove KMeans result from voting system
    y_pred_photo = np.zeros(len(X_test_sim_dist))  # Placeholder
    # Combine votes of other four methods
    votes_combined = y_pred_iso + y_pred_lof + y_pred_svm + y_pred_ae
    final_votes = np.array([-1 if vote < 0 else 1 for vote in votes_combined])
    # Calculate accuracy and classification report for voting of other four methods
    accuracy_votes = accuracy_score(y_test, final_votes)
    report_votes = classification_report(y_test, final_votes, target_names=['in', 'out'])
else:
    y_pred_photo = kmeans.predict(X_given_photo_scaled)

    # Get cluster centers
    centers = kmeans.cluster_centers_

    # Determine which cluster has smaller values based on 'vitb32_75_given_photo'
    if centers[0][0] < centers[1][0]:  # Compare the first feature of each cluster center
        smaller_cluster_label = 0
    else:
        smaller_cluster_label = 1

    # Convert cluster labels to -1 or 1 based on which cluster has smaller values
    y_pred_photo = np.where(y_pred_photo == smaller_cluster_label, -1, 1)

    # Combine votes of all five methods
    votes_combined = y_pred_iso + y_pred_lof + y_pred_svm + y_pred_ae + y_pred_photo
    final_votes = np.array([-1 if vote < 0 else 1 for vote in votes_combined])
    # Calculate accuracy and classification report for voting of all five methods
    accuracy_votes = accuracy_score(y_test, final_votes)
    report_votes = classification_report(y_test, final_votes, target_names=['in', 'out'])

print("Voting Accuracy:", accuracy_votes)
print("Classification Report for Voting:")
print(report_votes)
