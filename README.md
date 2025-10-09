# general-curriculum-global
SBERT-powered curriculum mapping: clustering university courses into semantic families and linking them to relevant journals.

## Overview

This project builds a semantic “general curriculum” map across 14 institutions to identify shared academic themes (“course families”) and connect them to the most relevant journals. It expands on the earlier department-level TF-IDF model by introducing semantic embeddings (SBERT) for concept-level similarity rather than surface-level keyword overlap.

## Objectives

• Group semantically similar courses across institutions into coherent “course families.”

• Infer fine-grained departments automatically (e.g., Psychology, Sociology, Political Science) without relying on predefined high-level categories.

• Link each course family to journals most closely aligned in subject matter.

• Deliver an interpretable Excel for Product and Sales teams showing curriculum coverage and journal alignment.

## Data

Input dataset (clean_text_global.xlsx) contains the following columns:

• institution: Institution name (14 total)

• course_id: Unique identifier for each course

• course_title: Course title (pre-deduplicated)

• course_description: Cleaned and preprocessed course text

• journal_title: Matched journal title

• journal_abstract: Preprocessed abstract text

• subject_area: subject collection label

## Methodology

### Embedding Generation

• Model: sentence-transformers/all-MiniLM-L6-v2

• Approach: Encode each course description and journal abstract into 384-dimensional embeddings.

• Normalization: All embeddings are L2-normalized for cosine similarity.

### Course Family Clustering

• Method: KMeans clustering on course embeddings.

• Hyperparameter: K = 50 (empirically chosen for balance between granularity and interpretability).

• Output: Each cluster represents a course family (e.g., “Social Psychology,” “Strategic Management”).

• Family names are inferred from representative course titles.

### Department Inference

• Each family centroid (mean embedding of its courses) is compared against SBERT embeddings of canonical department names.

• The top-2 most similar departments are assigned as primary and secondary fields.

• This allows automatic classification into domains like Psychology, Economics, Political Science, etc.

### Journal Mapping

• For each family, compute cosine similarity between its centroid and all journal embeddings.

• Select the Top 5 most semantically similar journals.

• This identifies which journals best support or align with the curricular themes.

## Deliverables

Output file: general_curriculum_global.xlsx

• Families_Summary: Each course family with inferred departments, #courses, #institutions

• Family_TopJournals: Top 5 journals per family (with similarity scores)

• Courses_to_Families: All courses and their assigned families

• Coverage_Pivot: Institution × Family matrix (for coverage heatmap)

## Key Insights

• SBERT captures conceptual similarity (e.g., “Social Psychology” and “Cognitive Psychology” cluster together even if titles differ).

• The approach scales easily across institutions and does not require manual labeling.

• Similarity scores between 0.6–0.75 indicate strong conceptual overlap between course families and journals.

## Tech Stack

• Language Model: Sentence-BERT (all-MiniLM-L6-v2)

• Clustering: KMeans (scikit-learn)

• Similarity Metric: Cosine similarity

• Visualization / Export: pandas + openpyxl

• Runtime Environment: Google Colab (GPU optional)

## Repository Structure
├── clean_text_global.xlsx
├── general_curriculum_global.xlsx
├── general_curriculum_global.ipynb
└── README.md

## Next Steps

• Add visualization layer (heatmap for coverage pivot, Sankey linking families <=> journals).

• Explore hierarchical clustering or topic modeling to refine large families.

• Integrate with metadata APIs to enrich journal context.
