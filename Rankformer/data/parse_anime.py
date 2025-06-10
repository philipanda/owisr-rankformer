import pandas as pd
import numpy as np
import pickle as pkl

random_state = 2137

# Load and reduce dataset
df = pd.read_csv('anime/rating.csv')
df = df.drop(columns=['rating'])

def continuous_missing(ids):
    return set(range(min(ids), max(ids))) - set(ids)

# to make the IDs continuous and start with 0
def remap_ids(df, col1, col2):
    map1 = {old: new for new, old in enumerate(sorted(df[col1].unique()))}
    map2 = {old: new for new, old in enumerate(sorted(df[col2].unique()))}
    df[col1] = df[col1].map(map1)
    df[col2] = df[col2].map(map2)
    mappings = {
        col1: map1,
        col2: map2
    }
    return df, mappings


df, id_mappings = remap_ids(df, 'user_id', 'anime_id')
with open("anime_id_mappings_dict.pkl", 'wb') as f:
    pkl.dump(id_mappings, f)

# Reduce amount of entries by around this extent
reduce_factor = 0.1

# removes up to the reduce_factor
unique_users = np.sort(df['user_id'].unique())
unique_users = unique_users[:int(len(unique_users) * reduce_factor)]
unique_items = np.sort(df['anime_id'].unique())
unique_items = unique_items[:int(len(unique_items) * reduce_factor)]
initial_len = len(df)
# restore all items/users that would could make the set not continuous
df = df[df['user_id'].isin(unique_users) & df['anime_id'].isin(unique_items)]
print("Reduced size percent:", len(df)/initial_len)

assert len(continuous_missing(unique_items))==0
assert len(continuous_missing(unique_users))==0

train_list, valid_list, test_list = [], [], []

# every user in every set, split using the interactions
for user_id, user_df in df.groupby('user_id'):
    user_df = user_df.sample(frac=1, random_state=random_state)
    n = len(user_df)

    train_end = int(n * 0.7)
    valid_end = int(n * 0.8)

    train_list.append(user_df.iloc[:train_end]) # 0.7
    valid_list.append(user_df.iloc[train_end:valid_end]) # 0.1
    test_list.append(user_df.iloc[valid_end:]) # 0.2

# random shuffle
train_df = pd.concat(train_list).sample(frac=1, random_state=random_state)
valid_df = pd.concat(valid_list).sample(frac=1, random_state=random_state)
test_df = pd.concat(test_list).sample(frac=1, random_state=random_state)

def write_to_file(df, filename):
    with open(filename, 'w') as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.user_id} {row.anime_id}\n")
    print(f"{filename}: {len(df)} rows")

write_to_file(train_df, 'anime/train.txt')
write_to_file(valid_df, 'anime/valid.txt')
write_to_file(test_df, 'anime/test.txt')

with open('anime/info.txt', 'w') as f:
    f.write(f"train {len(train_df)}\n")
    f.write(f"valid {len(valid_df)}\n")
    f.write(f"test {len(test_df)}\n")
    f.write(f"users {len(set(train_df['user_id']) | set(valid_df['user_id']) | set(test_df['user_id']))}\n")
    f.write(f"items {len(set(train_df['anime_id']) | set(valid_df['anime_id']) | set(test_df['anime_id']))}\n")
    f.write(f"--- {pd.Timestamp.now()} ---\n")
