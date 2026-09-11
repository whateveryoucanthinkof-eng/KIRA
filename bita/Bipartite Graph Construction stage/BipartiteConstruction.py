import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # Convert SourceIP and TargetIP to strings to ensure they are hashable
    df['SourceIP'] = df['SourceIP'].astype(str)
    df['TargetIP'] = df['TargetIP'].astype(str)

    # Initialize label encoders for categorical features
    proto_encoder = LabelEncoder()
    attack_type_encoder = LabelEncoder()
    port_type_encoder = LabelEncoder()

    # Fit and transform categorical features to numeric values
    df['Proto_encoded'] = proto_encoder.fit_transform(df['Proto'])
    df['Category_encoded'] = attack_type_encoder.fit_transform(df['Category'])
    df['Port_encoded'] = proto_encoder.fit_transform(df['Port'])

    # Map encoded categories back to their original labels
    category_mapping = {index: label for index, label in enumerate(attack_type_encoder.classes_)}

    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    # Encode SourceIP and TargetIP as integers
    df['SourceIP_encoded'], _ = pd.factorize(df['SourceIP'])
    df['TargetIP_encoded'], _ = pd.factorize(df['TargetIP'])

    for idx, row in df.iterrows():
        u = row['SourceIP_encoded']
        i = row['TargetIP_encoded']
        ts = row['DetectTime'].timestamp()
        label = row['Category_encoded']

        # Extracting FlowCount, Port, and Proto as features
        flow_count = row['FlowCount']
        port = row['Port_encoded']
        proto = row['Proto_encoded']
        attack_type = row['Category_encoded']

        #feat = np.zeros(172)
        feat = np.array([flow_count, port, proto])

        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)
        feat_l.append(feat)

    processed_df = pd.DataFrame({
        'u': u_list,
        'i': i_list,
        'ts': ts_list,
        'label': label_list,
        'idx': idx_list
    })
    # Adding a column to help identify the original date
    processed_df['original_date'] = pd.to_datetime(processed_df['ts'], unit='s')
    return processed_df, np.array(feat_l), category_mapping

def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique())), "u column not contiguous integers"
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique())), "i column not contiguous integers"

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

def run(df, bipartite=True):
    # Preprocess the DataFrame
    preprocessed_df, feat,category_mapping = preprocess(df)
    # Reindex the DataFrame
    new_df = reindex(preprocessed_df, bipartite)

    # Create empty feature for padding
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    # Create random features for nodes
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    # return the processed data
    return new_df, feat, rand_feat


new_df, feat, rand_feat = run(df, bipartite=True)
