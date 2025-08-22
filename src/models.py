"""
This script defines the Graph Neural Network (GNN) architectures used
in the DILI prediction project.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATClassifier(torch.nn.Module):
    """
    A Graph Attention Network (GAT) designed for classifying molecular graphs.
    This model was used as a feature extractor in our final hybrid model.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        """
        Initializes the GAT model layers.

        Args:
            num_node_features (int): The number of features for each node (atom).
            num_edge_features (int): The number of features for each edge (bond).
            hidden_channels (int): The number of neurons in the hidden layers.
        """
        super(GATClassifier, self).__init__()
        # First GAT layer with 2 attention heads
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=2, edge_dim=num_edge_features)
        # Second GAT layer, which consolidates the heads
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, edge_dim=num_edge_features)
        
        # Fully connected layers for classification
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def extract_embedding(self, data):
        """
        Processes a graph to generate a fixed-size embedding vector.
        This part of the forward pass can be used for feature extraction.

        Args:
            data (torch_geometric.data.Data): A batch of graph data.

        Returns:
            torch.Tensor: A graph-level embedding vector.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Pass through the GAT layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x) # Use ELU activation function
        x = self.conv2(x, edge_index, edge_attr)
        
        # Pool the node features to get a single vector for the whole graph
        embedding = global_mean_pool(x, batch)
        return embedding

    def forward(self, data):
        """
        The full forward pass for making a classification.

        Args:
            data (torch_geometric.data.Data): A batch of graph data.

        Returns:
            torch.Tensor: The final raw output (logit) for classification.
        """
        # Get the graph embedding
        embedding = self.extract_embedding(data)
        
        # Pass the embedding through the classification head
        x = F.relu(self.lin1(embedding))
        x = F.dropout(x, p=0.5, training=self.training) # Dropout for regularization
        x = self.lin2(x)
        
        return x