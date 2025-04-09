
| Stage                           | Criteria                                                                                                              | Included / Excluded | Paper Remaining Count |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification                   | Records identified using search terms across 7 CSV files from 5 databases                                            | Included            | 346                    |
| Deduplication (DOI link)         | Removed exact duplicates based on 'DOI link' field using Python script                                               | Excluded            | 340                    |
| Deduplication (Title)            | Removed additional duplicates using normalized 'Paper Title' with Python-based string cleaning and matching          | Excluded            | 337                    |
| AI, ML, DL Relevance Filter      | Filtered titles using Python to retain only papers mentioning AI, ML, or DL (handling hyphens, slashes, punctuation) | Excluded            | 140                    |
| Attention & Mental Health Filter | Filtered papers using Python to include only those mentioning attention, focus, cognitive load, or mental health     | Excluded            | 102                    |
| AI-Driven Method Filter          | Included only papers with AI-method terms (e.g., detection, prediction, modeling) in title using Python              | Included            | 84                     |
| Contextual/Digital Setting Filter| Included only papers with digital or behavioral context terms (e.g., learning, app, social media, human factors)     | Included            | 73                     |


## Stage Description: Contextual/Digital Setting Filter

In this final inclusion step, a Python script was used to retain only those papers whose titles mentioned contextual or digital environments relevant to AI-powered attention research. Each title was preprocessed by converting text to lowercase, replacing slashes and hyphens with spaces, and removing punctuation for consistency.

The script then matched the cleaned titles against an expanded list of keywords representing digital platforms, educational settings, and human interaction — including `"digital"`, `"learning"`, `"social media"`, `"application"`, `"user behavior"`, and `"human factors"`. This filtering ensured that only studies grounded in real-world digital contexts were kept.

The process refined the dataset from 84 to **73 papers**, yielding a highly targeted collection aligned with the theme: **AI-powered methods for assessing attention and focus in the digital age**.

