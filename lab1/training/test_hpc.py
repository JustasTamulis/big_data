import pandas as pd

# Load data
df = pd.read_csv("data/aisdk-small.csv")

# Group by mobile type and calculate mean width/height
result = (
    df.groupby("Type of mobile").agg({"Width": "mean", "Length": "mean"}).reset_index()
)

# Save to CSV
result.to_csv("output/mobile_dimensions.csv", index=False)
