
| Stage                      | Criteria                                                                                                      | Included / Excluded | Paper Remaining Count |
|---------------------------|---------------------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification            | Records identified using search terms across 7 CSV files from 5 databases                                     | Included            | 346                    |
| Deduplication (DOI link)  | Removed exact duplicates based on 'DOI link' field using Python script                                        | Excluded            | 340                    |
| Deduplication (Title)     | Removed additional duplicates using normalized 'Paper Title' with Python-based string cleaning and matching   | Excluded            | 337                    |
| AI, ML, DL Relevance Filter | Filtered titles using Python to retain only papers mentioning AI, ML, or DL (handling hyphens, slashes, and punctuation) | Excluded            | 140                    |




## **Stage Description: AI, ML, DL Relevance Filter**

In this stage, we developed a custom Python script to retain only the papers whose titles explicitly mentioned AI-related terms. The filtering logic was designed to be robust, accounting for common formatting patterns found in academic titles.

To accurately match keywords such as "AI", "ML", "DL", and their expanded forms, the program first normalized each title by:
- Converting all characters to lowercase.
- Replacing hyphens (`-`) and slashes (`/`) with spaces so that compound terms like `"AI-based"` and `"AI/ML"` become `"ai based"` and `"ai ml"`.
- Removing all other punctuation and reducing multiple spaces to single spaces.

The script then matched cleaned titles against a list of AI-related keywords using regular expressions. Only titles containing terms like `"ai"`, `"artificial intelligence"`, `"ml"`, `"machine learning"`, `"dl"`, or `"deep learning"` were retained.

This filtering step refined the dataset from 337 to 140 papers, ensuring high topical relevance for the subsequent screening and review stages.
