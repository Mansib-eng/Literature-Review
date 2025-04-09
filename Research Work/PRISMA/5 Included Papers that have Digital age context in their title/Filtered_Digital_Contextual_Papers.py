import pandas as pd
import re

# Load the previously filtered file
file_path = "Filtered_AI_Method_Papers.csv"
df = pd.read_csv(file_path)

# Expanded contextual/digital keywords
context_keywords = [
    # Digital platforms & media
    "digital", "online", "virtual", "mobile", "smartphone", "app", "application", "platform", "interface",
    "screen", "screen time", "web-based", "browser", "device",

    # Learning & education
    "learning", "education", "e-learning", "edtech", "learning system", "intelligent tutoring",
    "learning environment", "remote learning", "online class", "virtual classroom",

    # User behavior & interaction
    "user behavior", "behavior", "usage", "interaction", "engagement", "human factors",
    "user experience", "hci", "ui", "ux",

    # Social media & communication
    "social media", "social network", "media consumption", "digital communication",
    "online activity", "scrolling", "attention economy"
]

# Function to normalize and clean title text
def clean_text(text):
    text = str(text).lower()
    text = text.replace("/", " ").replace("-", " ")  # Handle slashes and hyphens
    text = re.sub(r'[^\w\s]', '', text)              # Remove punctuation
    text = re.sub(r'\s+', ' ', text)                 # Normalize whitespace
    return text.strip()

# Clean titles
df["Cleaned Title"] = df["Paper Title"].apply(clean_text)

# Compile regex pattern from keywords
pattern = r'\b(?:' + '|'.join(re.escape(word) for word in context_keywords) + r')\b'

# Filter titles that match at least one contextual keyword
filtered_df = df[df["Cleaned Title"].str.contains(pattern, regex=True)]

# Drop the helper column
filtered_df = filtered_df.drop(columns=["Cleaned Title"])

# Save the result
filtered_df.to_csv("Filtered_Contextual_Papers.csv", index=False)

# Print summary
print("Original papers:", len(df))
print("Filtered contextual/digital setting papers:", len(filtered_df))
print("Saved to: Filtered_Contextual_Papers.csv")
