

| Stage                      | Criteria                                                                                       | Included / Excluded | Paper Remaining Count |
|---------------------------|------------------------------------------------------------------------------------------------|---------------------|------------------------|
| Identification            | Records identified using search terms across 7 CSV files from 5 databases                      | Included            | 346                    |
| Deduplication (DOI link)  | Removed exact duplicates based on 'DOI link' field using automated Python script               | Excluded            | 340                    |
| Deduplication (Title)     | Removed additional duplicates using normalized 'Paper Title' via Python-based string matching  | Excluded            | 337                    |


---

**Identification**  
In the identification phase, records were gathered from seven CSV files, each representing exported results from literature searches conducted across five academic databases: IEEE Xplore, ACM Digital Library, ScienceDirect, Google Scholar, and PubMed. The searches were performed using keyword combinations related to "Artificial Intelligence", "Machine Learning", and "Attention/Focus". This stage resulted in 346 initial records.

**Deduplication (DOI Link)**  
To eliminate duplicate studies, an automated Python program was used to compare the "DOI link" field across all records. As each DOI uniquely identifies a publication, this method reliably removed redundant entries. This step excluded 6 duplicate records, reducing the dataset to 340 unique papers.

**Deduplication (Paper Title)**  
Some duplicate entries did not share identical DOIs due to formatting inconsistencies or missing values. To address this, the same Python program was extended to normalize the "Paper Title" field (by converting to lowercase, stripping whitespace, and standardizing spacing). A second deduplication step was then performed, removing an additional 3 records and resulting in a final dataset of 337 unique papers.

