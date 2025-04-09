import pandas as pd

# Desired columns after normalization
required_columns_map = {
    'searchterm': 'Search Term',
    'papertitle': 'Paper Title',
    'database': 'Database',
    'doilink': 'DOI link'
}

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

    # Normalize column names: lowercase and remove spaces
    df.columns = [col.strip().replace(" ", "").lower() for col in df.columns]

    # Rename to standard names using mapping
    df = df.rename(columns={key: value for key, value in required_columns_map.items() if key in df.columns})

    # Keep only the required 4 columns
    df = df[[required_columns_map[key] for key in required_columns_map if required_columns_map[key] in df.columns]]

    dataframes.append(df)

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined raw file
combined_df.to_csv("Combined_All_Papers.csv", index=False)

print("Original row count:", len(combined_df))

# Remove duplicates based on 'DOI link'
deduplicated_df = combined_df.drop_duplicates(subset="DOI link")

# Save the deduplicated version
deduplicated_df.to_csv("Combined_Deduplicated_Papers.csv", index=False)

print("Row count after removing duplicates:", len(deduplicated_df))
