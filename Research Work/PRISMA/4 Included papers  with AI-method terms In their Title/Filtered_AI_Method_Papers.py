import pandas as pd
import re

# Load the previous filtered dataset
file_path = "Filtered_Attention_MentalHealth_Papers.csv"
df = pd.read_csv(file_path)

# Expanded AI-driven method keywords
keywords = [
    "detection", "assessment", "tracking", "recognition", "estimation", "monitoring", "measurement", "prediction",
    "classification", "identification", "inference", "regression", "diagnosis", "analysis", "evaluation",
    "forecasting", "annotation", "scoring", "modeling", "tagging", "sensing", "quantification", "automation",
    "computation", "approximation", "labeling", "learning", "modeling technique", "data mining", "output generation"
]

# Normalize and clean title
def clean_text(text):
    text = str(text).lower()
    text = text.replace("/", " ").replace("-", " ")  # Normalize compound terms
    text = re.sub(r'[^\w\s]', '', text)              # Remove punctuation
    text = re.sub(r'\s+', ' ', text)                 # Normalize whitespace
    return text.strip()

# Apply title cleaning
df["Cleaned Title"] = df["Paper Title"].apply(clean_text)

# Create regex pattern
pattern = r'\b(?:' + '|'.join(re.escape(word) for word in keywords) + r')\b'

# Filter titles that match at least one keyword
filtered_df = df[df["Cleaned Title"].str.contains(pattern, regex=True)]

# Drop the helper column
filtered_df = filtered_df.drop(columns=["Cleaned Title"])

# Save filtered result
filtered_df.to_csv("Filtered_AI_Method_Papers.csv", index=False)

# Summary output
print("Original papers:", len(df))
print("Filtered papers (AI-driven methods):", len(filtered_df))
print("Saved to: Filtered_AI_Method_Papers.csv")
