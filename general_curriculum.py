!pip -q install sentence-transformers hdbscan openpyxl
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from google.colab import files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import hdbscan
from sentence_transformers import SentenceTransformer

uploaded = files.upload()
fname = list(uploaded.keys())[0]
print("Loaded file:", fname)

# Read Excel
df = pd.read_excel(fname)

# Verify expected schema
expected_cols = [
    "institution", "course_id", "course_title", "course_description",
    "journal_title", "journal_abstract", "subject_area"
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

df = df[expected_cols].copy()

# Light type coercion
for c in expected_cols:
    df[c] = df[c].astype(str)

# Quick sanity: counts & uniqueness
n_rows = len(df)
n_institutions = df["institution"].nunique()
n_courses = df[["institution","course_id"]].drop_duplicates().shape[0]
n_journals = df["journal_title"].nunique()

print(f"Rows: {n_rows}")
print(f"Institutions: {n_institutions}")
print(f"Unique courses: {n_courses}")
print(f"Unique journals: {n_journals}")

# Peek
df.head(5)

# Combine course title + description for richer semantics
course_texts = (df["course_title"].fillna("") + " || " +
                df["course_description"].fillna("")).tolist()

# Combine journal title + abstract
journal_texts = (df["journal_title"].fillna("") + " || " +
                 df["journal_abstract"].fillna("")).tolist()

# Load small but strong semantic model (fast, 384-dim)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode with normalization for cosine similarity
E_courses = model.encode(course_texts, batch_size=256, show_progress_bar=True,
                         normalize_embeddings=True)
E_journals = model.encode(journal_texts, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True)

print("Courses embedding shape:", E_courses.shape)
print("Journals embedding shape:", E_journals.shape)

### Run HDBSCAN (auto-K; tweak min_cluster_size if needed)
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
labels = clusterer.fit_predict(E_courses)

df["family_id"] = labels

# Build centroids for non-noise families
valid_mask = df["family_id"] >= 0
family_ids = sorted(df.loc[valid_mask, "family_id"].unique().tolist())

fam2idx = {fid: np.where(df["family_id"].values == fid)[0] for fid in family_ids}
fam_centroids = {fid: E_courses[idxs].mean(axis=0) for fid, idxs in fam2idx.items()}

# Reassign noise points (-1) to nearest centroid
if (df["family_id"] == -1).any() and len(family_ids) > 0:
    noise_idx = np.where(df["family_id"].values == -1)[0]
    C = np.stack([fam_centroids[f] for f in family_ids], axis=0)
    sims = cosine_similarity(E_courses[noise_idx], C)
    best = sims.argmax(axis=1)
    reassigned = np.array([family_ids[b] for b in best])
    df.loc[noise_idx, "family_id"] = reassigned

# Recompute families & centroids after reassignment (final)
family_ids = sorted(df["family_id"].unique().tolist())
fam2idx = {fid: np.where(df["family_id"].values == fid)[0] for fid in family_ids}
fam_centroids = {fid: E_courses[idxs].mean(axis=0) for fid, idxs in fam2idx.items()}

# Quick summary
family_sizes = pd.Series({fid: len(fam2idx[fid]) for fid in family_ids}).sort_values(ascending=False)
print(f"Total families: {len(family_ids)}")
print("Top 10 family sizes:\n", family_sizes.head(10))

### Cluster courses globally using KMeans (fixed K)

# Choose number of clusters (tune between 40–60)
K = 50
print(f"Running KMeans with K = {K} ...")

km = KMeans(n_clusters=K, n_init="auto", random_state=42)
labels = km.fit_predict(E_courses)
df["family_id"] = labels

# Build centroids
family_ids = sorted(df["family_id"].unique().tolist())
fam2idx = {fid: np.where(df["family_id"].values == fid)[0] for fid in family_ids}
fam_centroids = {fid: E_courses[idxs].mean(axis=0) for fid, idxs in fam2idx.items()}

# Quick summary
family_sizes = pd.Series({fid: len(fam2idx[fid]) for fid in family_ids}).sort_values(ascending=False)
print(f"Total families: {len(family_ids)}")
print("Top 10 family sizes:\n", family_sizes.head(10))

### Name each family (canonical title + summary)

"""Light cleanup for readable family names."""
def normalize_title(title: str) -> str:

    t = (title or "").strip()
    t_low = t.lower()

    # remove boilerplate openers
    t_low = re.sub(
        r'^\s*(introduction to|intro to|introduction|intro|topics in|special topics in|'
        r'selected topics in|research methods in|methods in|current topics in|current '
        r'literature in|seminar in|seminar on|advanced|capstone|workshop in)\s+', '', t_low)

    # drop subtitles after colon/dash
    t_low = re.sub(r'\s*[:\-–—].*$', '', t_low)

    # spacing & title case
    t_low = re.sub(r'\s+', ' ', t_low).strip()
    nice = t_low.title()

    # small fixes
    nice = re.sub(r'\bAnd\b', 'and', nice)
    return nice

family_rows = []
for fid in sorted(fam2idx.keys()):
    idxs = fam2idx[fid]
    centroid = fam_centroids[fid].reshape(1, -1)

    # choose course whose embedding is closest to the family centroid
    sims = cosine_similarity(centroid, E_courses[idxs]).ravel()
    best_idx = idxs[int(np.argmax(sims))]

    raw_name = df.loc[best_idx, "course_title"]
    clean_name = normalize_title(raw_name)

    n_courses = len(idxs)
    n_institutions = df.loc[idxs, "institution"].nunique()

    family_rows.append({
        "family_id": fid,
        "family_name": clean_name,
        "family_name_raw": raw_name,
        "n_courses": n_courses,
        "n_institutions": n_institutions
    })

FAMILIES = (pd.DataFrame(family_rows)
              .sort_values(["n_courses","n_institutions"], ascending=False)
              .reset_index(drop=True))

print("Families:", FAMILIES.shape)
FAMILIES.head(10)

### Infer fine-grained departments per family (semantic matching)

# Define target department taxonomy
target_departments = [
    # Social Sciences
    "Psychology", "Sociology", "Political Science", "Economics", "Anthropology", "Public Policy",
    # Business / Management
    "Accounting", "Finance", "Marketing", "Management", "Operations and Supply Chain", "Human Resources",
    "Entrepreneurship", "Business Analytics",
    # STEM / Quant
    "Computer Science", "Data Science", "Information Systems", "Engineering", "Mathematics", "Statistics",
    # Humanities / Other
    "Education", "Communication and Media Studies", "Geography", "Environmental Studies", "Law", "Health"
]

# Embed department labels
E_targets = model.encode(target_departments, normalize_embeddings=True)
T = np.stack(E_targets, axis=0)  # (#labels x dim)

# Build family centroids
# fam2idx and E_courses should already exist; recompute centroids just in case
fam_centroids = {fid: E_courses[idxs].mean(axis=0) for fid, idxs in fam2idx.items()}

# For each family, pick primary and secondary department
def infer_depts_for_family(fid: int, secondary_threshold: float = 0.35):
    c = fam_centroids[fid].reshape(1, -1)
    sims = cosine_similarity(c, T).ravel()
    order = sims.argsort()[::-1]
    primary = target_departments[order[0]]
    secondary = target_departments[order[1]] if sims[order[1]] >= secondary_threshold else ""
    return primary, secondary, float(sims[order[0]]), float(sims[order[1]])

dept_rows = []
for fid in sorted(fam2idx.keys()):
    p, s, sp, ss = infer_depts_for_family(fid)
    dept_rows.append({
        "family_id": fid,
        "primary_dept": p,
        "secondary_dept": s,
        "dept_sim_primary": sp,
        "dept_sim_secondary": ss
    })

DEPTS = pd.DataFrame(dept_rows)

# Merge onto FAMILIES table
FAMILIES = FAMILIES.merge(DEPTS, on="family_id", how="left")
print("FAMILIES with departments:", FAMILIES.shape)
FAMILIES.head(10)

### Link each course family to its top journals

TOPN_JOURNALS = 5  # change if want more per family

# Compute the journal embeddings once (if not already done)
E_journals = model.encode(df["journal_abstract"].tolist(), normalize_embeddings=True)

# Build a lookup of journal titles and subject areas
journals_meta = df[["journal_title", "subject_area"]].reset_index(drop=True)

# For each family, find the journals with highest cosine similarity to its centroid
journal_rows = []
for fid, centroid in fam_centroids.items():
    sims = cosine_similarity(centroid.reshape(1, -1), E_journals).ravel()
    top_idx = sims.argsort()[::-1][:TOPN_JOURNALS]

    for rank, j in enumerate(top_idx, start=1):
        journal_rows.append({
            "family_id": fid,
            "rank": rank,
            "journal_title": journals_meta.loc[j, "journal_title"],
            "journal_subject_area": journals_meta.loc[j, "subject_area"],
            "similarity": float(sims[j])
        })

FAMILY_JOURNALS = pd.DataFrame(journal_rows)
print("Family–Journal links:", FAMILY_JOURNALS.shape)
FAMILY_JOURNALS.head(10)

### Build deliverables & export to Excel

# Courses <=> Families mapping
COURSES_FAMILIES = (
    df[["institution","course_id","course_title","course_description","family_id"]]
    .merge(FAMILIES[["family_id","family_name","primary_dept","secondary_dept"]],
           on="family_id", how="left")
    .sort_values(["family_name","institution","course_title"])
)

# Coverage pivot (institution × family_name)
COVERAGE_PIVOT = (
    COURSES_FAMILIES.assign(present=1)
    .pivot_table(index="institution", columns="family_name", values="present",
                 aggfunc="sum", fill_value=0)
    .reset_index()
)

# Families summary (already contains counts + inferred depts)
FAMILIES_SUMMARY = FAMILIES[[
    "family_id","family_name","primary_dept","secondary_dept",
    "n_courses","n_institutions","dept_sim_primary","dept_sim_secondary"
]].sort_values(["n_courses","n_institutions"], ascending=False)

# Family => Top journals
FAMILY_TOPJOURNALS = FAMILY_JOURNALS.copy().sort_values(["family_id","rank"])

# Export
outname = "general_curriculum_global.xlsx"
with pd.ExcelWriter(outname, engine="openpyxl") as writer:
    FAMILIES_SUMMARY.to_excel(writer, sheet_name="Families_Summary", index=False)
    FAMILY_TOPJOURNALS.to_excel(writer, sheet_name="Family_TopJournals", index=False)
    COURSES_FAMILIES.to_excel(writer, sheet_name="Courses_to_Families", index=False)
    COVERAGE_PIVOT.to_excel(writer, sheet_name="Coverage_Pivot", index=False)

print("Saved:", outname)
print({
    "Families_Summary": FAMILIES_SUMMARY.shape,
    "Family_TopJournals": FAMILY_TOPJOURNALS.shape,
    "Courses_to_Families": COURSES_FAMILIES.shape,
    "Coverage_Pivot": COVERAGE_PIVOT.shape
})

files.download(outname)
