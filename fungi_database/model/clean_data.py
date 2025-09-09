#-------------------------------------------------------------------------------
# Name:        fungi_model_data_preprocessing
# Purpose:
#
# Author:      bluer
#
# Created:     18/08/2025
# Copyright:   (c) bluer 2025
# Licence:     MIT
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

#Load dataset
df = pd.read_csv(r"C:\Users\holly\Documents\projects\fungi_database\examples\joined_fungi_metadata.csv", low_memory=False)
#Define non-fungus columns
metaData_cols = ["sample_ID", "paper_ID", "paper_title", "paper_year", "paper_authors", "paper_journal",
    "paper_doi", "paper_sample_name", "latitude", "longitude", "elevation_study", "continent", "country",
    "location", "year_of_sampling_from", "year_of_sampling_to", "month_of_sampling", "day_of_sampling",
    "sample_type", "sample_type_specification", "environment_type", "ecosystem_classification", "dominant_plant_species",
    "other_plant_species", "manipulated", "experimental_manipulation_type", "experimental_manipulation_direction",
     "experimental_manipulation_vegetation", "experimental_manipulation_duration", "experimental_manipulation_frequency",
     "experimental_manipulation_application_detail", "experimental_manipulation_intensity", "pH", "pH_method",
     "organic_matter_content", "organic_C_content", "total_N_content", "total_Ca", "total_P", "total_K", "MAT_study",
     "MAP_study", "area_GPS", "area_sampled", "number_of_subsamples", "sampling_info", "sample_depth", "sample_info",
     "DNA_extraction_sample_mass", "DNA_extraction_size", "DNA_extraction_method", "barcoding_region", "PCR_primers",
     "PCR_primers_sequence", "sequencing_platform", "ITS1_extracted", "ITS2_extracted", "ITS_total", "date_added",
     "submitted_by"]
#Missing values subbed as zero
abundance_cols = [col for col in df.columns if col not in metaData_cols]

df[abundance_cols] = df[abundance_cols].apply(pd.to_numeric, errors="coerce")
bad_counts = df[abundance_cols].isna().sum().sum()
if bad_counts > 0:
    print(f"Warning: {bad_counts} non-numeric entries found in abundance data. Replaced with zero.")
df[abundance_cols] = df[abundance_cols].fillna(0)

#Flag required data cols
critical_meta = ["latitude", "longitude", "dominant_plant_species", "pH", "MAT_study", "MAP_study"]

#Flag and count samples with missing values
df["missing_count"] = df[critical_meta].isnull().sum(axis=1)
print("\nMissing counts per sample:")
print(df["missing_count"].value_counts().sort_index())
print("\nMissing values per column:")
print(df[critical_meta].isnull().sum())
print("\nPercent missing per column:")
print((df[critical_meta].isnull().mean()*100).round(2))
summary = pd.DataFrame({
    "missing_count": df["missing_count"].value_counts().sort_index()
})
#Export summary of missing values
summary.to_csv("missing_counts_summary.csv", index=True)

#Filter out samples with <1 missing values
df_filtered = df[df["missing_count"]<=2].copy()
print(f"\nOriginal dataset size: {len(df)}")
print(f"Filtered dataset size: {len(df_filtered)}")
print("Dropped samples saved to 'fungi_db_samples_dropped.csv'")

#Standardize categories
df["dominant_plant_species"] = (
    df["dominant_plant_species"]
    .astype(str)
    .str.strip()
    .str.capitalize()
    .replace({
    "q. robur": "Quercus robur",
    "quercus robur": "Quercus robur"
    })
)
df["ecosystem_classification"] =(
    df["ecosystem_classification"]
    .astype(str)
    .str.lower()
    .replace({
    "woodland": "forest",
    "rain forest": "forest",
    "grass land": "grassland"
    })
)
#Convert strings to num

def clean_elevation(col: pd.Series, special_map=None) -> pd.Series:
    """
    Clean an elevation column:
    - Convert to numeric where possible
    - Map special text values to meaningful numbers or NaN
    """
    # Make everything lowercase for matching
    col_str = col.astype(str).str.lower().str.strip()

    # Map common special cases
    replacements = {
        "na": np.nan,
        "nan": np.nan,
        "none": np.nan,
        "missing": np.nan,
        "unknown": np.nan,
        "sea level": 0,
        "0m": 0,
        "0 meters": 0,
        "below sea level": 0,  # adjust if needed
    }
    if special_map:
        replacements.update(special_map)
    col_str = col_str.replace(replacements)
    return pd.to_numeric(col_str, errors="coerce")
#Clean numeric columns
numeric_meta_cols = ["elevation_study", "latitude", "longitude", "pH", "MAT_study", "MAP_study"]
special_cases = {
    "elevation_study":{ "sea level": 0, "0m": 0, "below sea level": 0}
}
for col in numeric_meta_cols:
    df[col] = clean_elevation(df[col], special_map=special_cases.get(col))

#Convert units
if df["elevation_study"].max() >9000:
    df["elevation_study"] = df["elevation_study"]*0.3048
if df["MAT_study"] .mean() >100:
    df["MAT_study"] = df["MAT_study"] - 273.15
if df["MAP_study"].mean() <100:
    df["MAP_study"] = df["MAP_study"] *10

print("Converting elevation from feet to meters...")
df["elevation_study"] = df["elevation_study"]*0.3048

#Flag outliers
df["flag_outliers"] = (
    (df["pH"]<0) | (df["pH"]>14) |
    (df["latitude"] <-90) | (df["latitude"] > 90) |
    (df["longitude"] <-180) | (df["longitude"] >180) |
    (df["MAP_study"] <0) | (df["MAT_study"] <-50)
    )
print(f"\n Flagged {df['flag_outliers'].sum()} rows as potential outliers.")

#Drop unneeded cols
drop_cols = [
    "paper_authors", "submitted_by", "ITS_total", "ITS2_extracted", "ITS1_extracted",
    "sequencing_platform", "PCR_primers_sequence", "PCR_primers", "barcoding_region",
    "DNA_extraction_method", "DNA_extraction_size", "DNA_extraction_sample_mass",
    "sampling_info", "number_of_subsamples", "area_sampled", "pH_method", "experimental_manipulation_intensity",
    "experimental_manipulation_application_detail", "experimental_manipulation_frequency", "experimental_manipulation_duration",
    "experimental_manipulation_direction", "experimental_manipulation_vegetation", "experimental_manipulation_type",
    "sample_type_specification", "location", "country", "continent", "paper_sample_name", "paper_journal", "paper_year",
    "paper_title", "paper_ID"
]
df = df.drop(columns = [c for c in drop_cols if c in df.columns])

#Save cleaned csv
os.makedirs("cleaned", exist_ok=True)
df.to_csv("fungi_clean.csv", index=False)
print(r"Cleaned and reordered dataset saved to 'fungi_clean.csv'")