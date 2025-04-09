
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



##  Manual Title Screening

In the final filtering stage, a manual review was conducted on the 73 papers that passed all previous automated filters. Each paper title was carefully examined to determine whether it directly aligned with the core focus of the systematic review: **AI-powered methods for assessing attention and focus in digital environments**.

Papers that were off-topic, too general, focused on unrelated AI applications, or lacked clear connection to cognitive attention were excluded based on expert judgment. This subjective title screening ensured that only the most relevant and focused studies were retained.

As a result, the dataset was refined to a final set of **38 papers**, which were deemed suitable for abstract screening and full-text review in the next stages of the systematic review process.
