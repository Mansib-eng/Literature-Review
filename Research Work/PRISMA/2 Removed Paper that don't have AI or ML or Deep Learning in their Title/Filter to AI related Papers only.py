import pandas as pd
import re

# Load the deduplicated file
file_path = "Combined_Deduplicate_removed_By_DOI_and_Title.csv"
df = pd.read_csv(file_path)

# Define AI-related keywords to check for
ai_keywords = ["ai", "artificial intelligence", "ml", "machine learning", "dl", "deep learning"]

# Function to clean and normalize text
def clean_text(text):
    text = str(text).lower()  # convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return text.strip()

# Clean the paper titles
df["Cleaned Title"] = df["Paper Title"].apply(clean_text)

# Build a pattern to match any AI-related keyword
pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in ai_keywords) + r')\b'

# Filter rows that contain any AI keyword in the cleaned title
filtered_df = df[df["Cleaned Title"].str.contains(pattern, regex=True)]

# Drop the helper column
filtered_df = filtered_df.drop(columns=["Cleaned Title"])

# Save the result
filtered_df.to_csv("Filtered_AI_Relevant_Papers.csv", index=False)

# Print summary
print("Original total papers:", len(df))
print("Filtered AI-relevant papers:", len(filtered_df))
print("Saved to: Filtered_AI_Relevant_Papers.csv")
