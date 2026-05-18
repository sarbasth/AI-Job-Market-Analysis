import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("ai_jobs_global.csv")

print(df.head())

# ======================
# REMOVE DUPLICATES
# ======================
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["job_title", "company", "city", "posted_date"])

# ======================
# HANDLE MISSING VALUES
# ======================
df = df.dropna(subset=["salary_min", "salary_max"])

df["city"] = df["city"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["remote_type"] = df["remote_type"].fillna("Unspecified")

# ======================
# CLEAN LOCATIONS (IMPORTANT FIX)
# ======================
# Remove unspecified remote types
df = df[df["remote_type"] != "Unspecified"]
df["city"] = df["city"].astype(str).str.strip().str.title()

df["country"] = df["country"].astype(str).str.strip().str.upper()

df["country"] = df["country"].replace({
    "Uk": "UK",
    "Us": "US"
})

# remove bad locations BEFORE analysis
bad_locations = ["The City", "Grand Central", "Unknown", "N/A", "None", "Uk", "Us"]
df = df[~df["city"].isin(bad_locations)]

# ======================
# SALARY FEATURE
# ======================
df["salary_avg"] = (df["salary_min"] + df["salary_max"]) / 2
df = df[df["salary_avg"] > 0]

# ======================
# BASIC INSIGHTS
# ======================
print("\n💷 Average salary:")
print(df["salary_avg"].mean())

print("\n📍 Top cities:")
print(df["city"].value_counts().head(10))

print("\n🏢 Top companies:")
print(df["company"].value_counts().head(10))

print("\n🏠 Remote split:")
print(df["remote_type"].value_counts())

print("\n🧑‍💻 Experience salary:")
print(df.groupby("experience_level")["salary_avg"].mean())

# ======================
# VISUALS
# ======================

# ======================
# CHART STYLE
# ======================
plt.style.use("dark_background")
sns.set(style="darkgrid")

# ======================
# 1. SALARY DISTRIBUTION (HISTOGRAM)
# ======================
plt.figure(figsize=(12,6))

sns.histplot(
    df["salary_avg"],
    bins=30,
    kde=True
)

plt.title("Salary Distribution", fontsize=18)
plt.xlabel("Average Salary")
plt.ylabel("Count")

plt.show()

# ======================
# 2. TOP SKILLS (BAR CHART)
# ======================
skills = (
    df["required_skills"]
    .dropna()
    .str.lower()
    .str.split(",")
    .explode()
    .str.strip()
)

top_skills = skills.value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_skills.values,
    y=top_skills.index,
    palette="Blues_r"
)

plt.title("Top Skills in Demand", fontsize=18)
plt.xlabel("Demand Count")
plt.ylabel("Skills")

plt.show()

# ======================
# 3. TOP LOCATIONS (BAR CHART)
# ======================
top_locations = df["city"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_locations.values,
    y=top_locations.index,
    palette="Blues_r"
)

plt.title("Top Job Locations", fontsize=18)
plt.xlabel("Number of Jobs")
plt.ylabel("City")

plt.show()

# ======================
# 4. REMOTE WORK % (BAR CHART)
# ======================
remote_counts = df["remote_type"].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(
    x=remote_counts.index,
    y=remote_counts.values,
    palette="Blues_r"
)

plt.title("Remote Work Distribution", fontsize=18)
plt.xlabel("Remote Type")
plt.ylabel("Count")

plt.show()

# ======================
# 5. HEATMAP
# ======================
plt.figure(figsize=(6,4))

sns.heatmap(
    df[["salary_min", "salary_max", "salary_avg"]].corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Salary Correlation Heatmap", fontsize=16)

plt.show()

df.to_csv("cleaned_ai_jobs.csv", index=False)