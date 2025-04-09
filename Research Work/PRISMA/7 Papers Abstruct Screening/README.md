
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

---

## Abstract Screening

In this stage, the abstracts of the remaining 38 papers were manually reviewed to determine their **high relevance** to the research topic:  
**“A Systematic Review on AI-Powered Methods for Assessing Attention and Focus in the Digital Age.”**

Each abstract was evaluated against five core research questions, focusing on the presence of:
- **AI/ML techniques** applied to **attention or focus assessment**
- Use of relevant **datasets and metrics**
- Evidence of **model effectiveness**
- Discussion of **technical challenges**
- Suggestions for **future directions**

Papers that clearly addressed these dimensions in digital environments (e.g., online learning, smart classrooms, real-time monitoring systems) were retained. The process resulted in **25 final papers** selected for full-text review and data extraction. This ensures a high-quality dataset that directly supports the objectives of the systematic literature review.
