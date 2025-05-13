
| Stage                           | Criteria                                                                                                              | Included / Excluded | Paper Remaining Count |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification                   | Records identified using search terms across 7 CSV files from 5 databases                                            | Included            | 346                    |
| Deduplication (DOI link)         | Removed exact duplicates based on 'DOI link' field using Python script                                               | Excluded            | 340                    |
| Deduplication (Title)            | Removed additional duplicates using normalized 'Paper Title' with Python-based string cleaning and matching          | Excluded            | 337                    |
| AI, ML, DL Relevance Filter      | Filtered titles using Python to retain only papers mentioning AI, ML, or DL (handling hyphens, slashes, punctuation) | Excluded            | 140                    |
| Attention & Mental Health Filter | Filtered papers using Python to include only those mentioning attention, focus, cognitive load, or mental health     | Excluded            | 102                    |
| AI-Driven Method Filter          | Included only papers with AI-method terms (e.g., detection, prediction, modeling) in title using Python              | Included            | 84                     |
| Contextual/Digital Setting Filter| Included only papers with digital or behavioral context terms (e.g., learning, app, social media, human factors)     | Included            | 73                     |
| Manual Title Screening           | Manually excluded irrelevant papers based on subjective review of titles and topic fit                               | Excluded            | 38                     |
| Abstract Screening               | Retained only highly relevant papers after manual review of abstracts against SLR criteria and RQs                   | Included            | 25                     |
| PDF Retrieval                    | Successfully retrieved full-text PDF versions of selected papers for in-depth review                                  | Included            | 19                     |



---


# PDF Retrieval  :

Out of the 25 papers selected after abstract screening, full-text PDFs for 19 papers have been successfully retrieved. These documents will be used for the final synthesis, ensuring that a majority of the selected studies can be analyzed in-depth for methodology, results, and relevance to the systematic review.








