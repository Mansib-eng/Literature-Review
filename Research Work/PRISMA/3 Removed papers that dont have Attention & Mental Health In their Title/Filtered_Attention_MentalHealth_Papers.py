import pandas as pd
import re

# Load the file
file_path = "Filtered_AI_Relevant_Papers.csv"
df = pd.read_csv(file_path)

# Combined keyword list
keywords = [
    # Attention & Cognitive Terms
    "attention", "focus", "concentration", "cognitive load", "mental effort", "engagement", "vigilance",
    "alertness", "inattention", "sustained attention", "selective attention", "divided attention",
    "cognitive effort", "attention span", "attentional control", "attentional state", "distraction",
    "mind wandering", "task performance", "mental workload", "information overload", "executive function",
    "attentional fatigue", "attentional capacity", "attentional shift", "working memory",
    "psychological state", "user attention", "cognitive attention", "attention level", "attentional behavior",

    # Mental Health–Related Terms
    "adhd", "add", "autism", "asd", "anxiety", "depression", "neurodivergent", "neurodevelopmental",
    "cognitive disorder", "mental health", "behavioral disorder", "executive dysfunction",
    "emotional regulation", "learning disability", "psychiatric disorder", "attention disorder",
    "cognitive impairment", "neurocognitive", "mental state"
]

# Normalize text function
def clean_text(text):
    text = str(text).lower()
    text = text.replace("/", " ").replace("-", " ")  # handle AI/ML, AI-based
    text = re.sub(r'[^\w\s]', '', text)              # remove punctuation
    text = re.sub(r'\s+', ' ', text)                 # normalize whitespace
    return text.strip()

# Apply cleaning to titles
df["Cleaned Title"] = df["Paper Title"].apply(clean_text)

# Build a regex pattern for all keywords
pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'

# Filter titles that contain at least one keyword
filtered_df = df[df["Cleaned Title"].str.contains(pattern, regex=True)]

# Drop helper column
filtered_df = filtered_df.drop(columns=["Cleaned Title"])

# Save the final filtered set
filtered_df.to_csv("Filtered_Attention_MentalHealth_Papers.csv", index=False)

# Print summary
print("Original papers:", len(df))
print("Filtered papers (attention + mental health terms):", len(filtered_df))
print("Saved to: Filtered_Attention_MentalHealth_Papers.csv")
