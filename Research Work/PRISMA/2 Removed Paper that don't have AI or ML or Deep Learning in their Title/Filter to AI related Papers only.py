import pandas as pd
import re

# Load the deduplicated CSV
file_path = "Combined_Deduplicate_removed_By_DOI_and_Title.csv"
df = pd.read_csv(file_path)

# AI-related keywords to detect
ai_keywords = ["ai", "artificial intelligence", "ml", "machine learning", "dl", "deep learning"]

# Normalize and clean titles
def clean_text(text):
    text = str(text).lower()
    text = text.replace("/", " ")   # handle AI/ML → ai ml
    text = text.replace("-", " ")   # handle AI-based → ai based
    text = re.sub(r'[^\w\s]', '', text)  # remove all other punctuation
    text = re.sub(r'\s+', ' ', text)     # normalize whitespace
    return text.strip()

# Apply cleaning
df["Cleaned Title"] = df["Paper Title"].apply(clean_text)

# Regex pattern for any AI keyword
pattern = r'\b(?:' + '|'.join(re.escape(word) for word in ai_keywords) + r')\b'

# Filter rows containing any AI keyword
filtered_df = df[df["Cleaned Title"].str.contains(pattern, regex=True)]

# Drop helper column
filtered_df = filtered_df.drop(columns=["Cleaned Title"])

# Save result
filtered_df.to_csv("Filtered_AI_Relevant_Papers.csv", index=False)

# Print results
print("Original total papers:", len(df))
print("Filtered AI-relevant papers:", len(filtered_df))
print("Saved to: Filtered_AI_Relevant_Papers.csv")
