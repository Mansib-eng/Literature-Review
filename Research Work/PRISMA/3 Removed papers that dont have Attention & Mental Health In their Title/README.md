
| Stage                           | Criteria                                                                                                              | Included / Excluded | Paper Remaining Count |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification                   | Records identified using search terms across 7 CSV files from 5 databases                                            | Included            | 346                    |
| Deduplication (DOI link)         | Removed exact duplicates based on 'DOI link' field using Python script                                               | Excluded            | 340                    |
| Deduplication (Title)            | Removed additional duplicates using normalized 'Paper Title' with Python-based string cleaning and matching          | Excluded            | 337                    |
| AI, ML, DL Relevance Filter      | Filtered titles using Python to retain only papers mentioning AI, ML, or DL (handling hyphens, slashes, punctuation) | Excluded            | 140                    |
| Attention & Mental Health Filter | Filtered papers using Python to include only those mentioning attention, focus, cognitive load, or mental health     | Excluded            | 102 |



### **Stage Description: Attention & Mental Health Filter**

In this stage, we applied a Python script to retain only those papers whose titles are semantically aligned with the core research objective: AI-powered methods for assessing attention and mental focus.

To ensure comprehensive matching, the script used a curated list of over 50 keywords related to **attention**, **cognitive function**, and **mental health conditions** such as "attention span", "cognitive load", "ADHD", and "working memory". The program normalized each title by converting to lowercase, replacing punctuation (including slashes and hyphens) with spaces, and removing extraneous symbols.

Regular expressions were used to ensure accurate keyword matching. Only papers containing at least one of these domain-specific terms in their titles were retained. This resulted in a further refined dataset highly relevant to the intended scope of the systematic review.
