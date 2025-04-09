
| Stage                           | Criteria                                                                                                              | Included / Excluded | Paper Remaining Count |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification                   | Records identified using search terms across 7 CSV files from 5 databases                                            | Included            | 346                    |
| Deduplication (DOI link)         | Removed exact duplicates based on 'DOI link' field using Python script                                               | Excluded            | 340                    |
| Deduplication (Title)            | Removed additional duplicates using normalized 'Paper Title' with Python-based string cleaning and matching          | Excluded            | 337                    |
| AI, ML, DL Relevance Filter      | Filtered titles using Python to retain only papers mentioning AI, ML, or DL (handling hyphens, slashes, punctuation) | Excluded            | 140                    |
| Attention & Mental Health Filter | Filtered papers using Python to include only those mentioning attention, focus, cognitive load, or mental health     | Excluded            | 102                    |
| AI-Driven Method Filter          | Included only papers with AI-method terms (e.g., detection, prediction, modeling) in title using Python              | Included            | 84                     |



##  AI-Driven Method Filter**

In this stage, a Python script was used to further refine the dataset by **including only those papers** whose titles mentioned AI-driven methodological terms. The purpose was to ensure that the retained studies not only addressed attention or mental health but also applied concrete AI techniques in their approach.

To achieve this, the script normalized each title by converting text to lowercase, replacing hyphens and slashes with spaces (to correctly detect phrases like "AI/ML" or "model-based"), and removing all punctuation. A comprehensive list of AI method keywords was used, including terms such as `"detection"`, `"assessment"`, `"modeling"`, `"learning"`, `"prediction"`, and `"classification"`. Regular expressions with word boundary matching ensured accurate filtering without capturing partial words.

As a result, **84 papers were included** in the final dataset, all of which explicitly reference AI methodologies relevant to attention, cognitive analysis, or mental health contexts.

