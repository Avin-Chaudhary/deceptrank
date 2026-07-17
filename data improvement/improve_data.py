import pandas as pd
from datetime import datetime

# Load CSV
df = pd.read_csv("interactions.csv")

# Step 1: Convert veracity
df['veracity'] = df['veracity'].map({
    'rumours': 'fake',
    'non-rumours': 'real'
}).fillna('unverified')

# Step 2: Convert timestamp to UNIX
def convert_to_unix(ts):
    dt = datetime.strptime(ts, "%a %b %d %H:%M:%S %z %Y")
    return int(dt.timestamp())

df['timestamp'] = df['timestamp'].apply(convert_to_unix)

# Step 3: Get unique interaction types
unique_types = df['interaction_type'].unique()
print("Unique interaction types:", unique_types)

# Save updated file
df.to_csv("output.csv", index=False)

print("✅ Done! Saved as output.csv")