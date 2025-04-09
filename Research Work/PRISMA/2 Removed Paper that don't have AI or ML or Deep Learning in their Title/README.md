
| Stage                      | Criteria                                                                                                      | Included / Excluded | Paper Remaining Count |
|---------------------------|---------------------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification            | Records identified using search terms across 7 CSV files from 5 databases                                     | Included            | 346                    |
| Deduplication (DOI link)  | Removed exact duplicates based on 'DOI link' field using Python script                                        | Excluded            | 340                    |
| Deduplication (Title)     | Removed additional duplicates using normalized 'Paper Title' with Python-based string cleaning and matching   | Excluded            | 337                    |
| AI, ML, DL Relevance Filter | Filtered titles using Python to retain only papers mentioning AI, ML, or DL (case-insensitive, cleaned)        | Excluded            | 140                    |



## AI, ML, DL Relevance Filter

In this stage, a Python script was used to filter out papers that did not mention AI, ML, DL, or related terms in their titles. The titles were normalized by removing punctuation, converting to lowercase, and trimming whitespace. Only papers with relevant keywords such as "AI", "Artificial Intelligence", "ML", "Machine Learning", "DL", or "Deep Learning" were retained. This ensured that the final dataset focused strictly on AI-powered methods, reducing the total to 140 papers.
