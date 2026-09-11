import torch.nn as nn

## Edge Features
# Convert categorical features to indices
flow_count_mapping = {category: idx for idx, category in enumerate(df['FlowCount'].unique())}
proto_mapping = {category: idx for idx, category in enumerate(df['Proto'].unique())}
category_mapping = {category: idx for idx, category in enumerate(df['Category'].unique())}

df['FlowCount_idx'] = df['FlowCount'].map(flow_count_mapping)
df['Proto_idx'] = df['Proto'].map(proto_mapping)
df['Category_idx'] = df['Category'].map(category_mapping)

# Define embedding size
embedding_dim = 4  # embedding size

# Define embedding layers
flow_count_embedding = nn.Embedding(len(flow_count_mapping), embedding_dim)
proto_embedding = nn.Embedding(len(proto_mapping), embedding_dim)
category_embedding = nn.Embedding(len(category_mapping), embedding_dim)

# Create embeddings for edge features
flow_count_embedded = flow_count_embedding(torch.tensor(df['FlowCount_idx'].values))
proto_embedded = proto_embedding(torch.tensor(df['Proto_idx'].values))
category_embedded = category_embedding(torch.tensor(df['Category_idx'].values))

# Combine embeddings into a single array
edge_features = torch.cat((flow_count_embedded, proto_embedded, category_embedded), dim=1).detach().numpy()

## Node Features
# Create mappings for categorical features
port_mapping = {category: idx for idx, category in enumerate(df['Port'].unique())}
proto_mapping = {category: idx for idx, category in enumerate(df['Proto'].unique())}
category_mapping = {category: idx for idx, category in enumerate(df['Category'].unique())}

# Map categorical features to indices
df['Port_idx'] = df['Port'].map(port_mapping)
df['Proto_idx'] = df['Proto'].map(proto_mapping)
df['Category_idx'] = df['Category'].map(category_mapping)

# Define embedding size
embedding_dim = 4  # Example embedding size

# Define embedding layers
port_embedding = nn.Embedding(len(port_mapping), embedding_dim)
proto_embedding = nn.Embedding(len(proto_mapping), embedding_dim)
category_embedding = nn.Embedding(len(category_mapping), embedding_dim)

# Convert indices to tensors
port_idx_tensor = torch.tensor(df['Port_idx'].values, dtype=torch.long)
proto_idx_tensor = torch.tensor(df['Proto_idx'].values, dtype=torch.long)
category_idx_tensor = torch.tensor(df['Category_idx'].values, dtype=torch.long)

# Create embeddings for node features
port_embedded = port_embedding(port_idx_tensor)
proto_embedded = proto_embedding(proto_idx_tensor)
category_embedded = category_embedding(category_idx_tensor)

# Combine embeddings into a single array
node_features = torch.cat((port_embedded, proto_embedded, category_embedded), dim=1).detach().numpy()
