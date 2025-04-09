import pandas as pd

file_paths = [
    "1. AI-ML Focused (Generic).csv",
    "2 AI or ML + Human Factors.csv",
    "3 Computer Science + Non-AI Technical Methods.csv",
    "4 AI + Technology Usage.csv",
    "5. Tech Usage Only (No AI).csv",
    "6. Education-Learning Environments + AI.csv",
    "7 Healthcare or Mental Health + AI.csv"
]

dataframes = []

for file in file_paths:
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='ISO-8859-1')
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

# Save combined (raw) file before removing duplicates
combined_df.to_csv("Combined_All_Papers.csv", index=False)

print("Original row count:", len(combined_df))

deduplicated_df = combined_df.drop_duplicates(subset="DOI link")
print("Row count after removing duplicates:", len(deduplicated_df))

deduplicated_df.to_csv("Combined_Deduplicated_Papers.csv", index=False)
