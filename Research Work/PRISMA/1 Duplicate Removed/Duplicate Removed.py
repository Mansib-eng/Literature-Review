import pandas as pd

# Correct file list
file_paths = [
    "1. AI-ML Focused (Generic).csv",
    "2 AI or ML + Human Factors.csv",
    "3 Computer Science + Non-AI Technical Methods.csv",
    "4 AI + Technology Usage.csv",
    "5. Tech Usage Only (No AI).csv",
    "6. Education-Learning Environments + AI.csv",
    "7 Healthcare or Mental Health + AI.csv"
]

# Column normalization mapping
required_columns_map = {
    'searchterm': 'Search Term',
    'papertitle': 'Paper Title',
    'database': 'Database',
    'doilink': 'DOI link'
}

dataframes = []

for file in file_paths:
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='ISO-8859-1')

    # Normalize columns
    df.columns = [col.strip().replace(" ", "").lower() for col in df.columns]
    df = df.rename(columns={key: value for key, value in required_columns_map.items() if key in df.columns})

    # Keep only required columns
    df = df[[required_columns_map[key] for key in required_columns_map if required_columns_map[key] in df.columns]]
    dataframes.append(df)

# Step 1: Combine all
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv("Combined_All_Papers.csv", index=False)
print("Combined rows:", len(combined_df))

# Step 2: Remove duplicates by DOI link
dedup_by_doi = combined_df.drop_duplicates(subset="DOI link").copy()
dedup_by_doi.to_csv("Combined_Deduplicate_removed_By_DOI.csv", index=False)
print("Rows after removing DOI duplicates:", len(dedup_by_doi))

# Step 3: Remove duplicates by cleaned Paper Title
dedup_by_doi["Cleaned Title"] = dedup_by_doi["Paper Title"].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
final_df = dedup_by_doi.drop_duplicates(subset="Cleaned Title").drop(columns=["Cleaned Title"])
final_df.to_csv("Combined_Deduplicate_removed_By_DOI_and_Title.csv", index=False)
print("Rows after removing Paper Title duplicates:", len(final_df))
