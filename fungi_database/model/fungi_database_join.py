#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      holly
#
# Created:     05/08/2025
# Copyright:   (c) holly 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd

fungi = r"C:\Users\holly\Documents\projects\fungi_database\GlobalFungi_5_species_abundance_ITS1_ITS2.txt\GlobalFungi_5_species_abundance_ITS1_ITS2.csv"
fungi_metadata = r"C:\Users\holly\Documents\projects\fungi_database\GlobalFungi_5_sample_metadata.txt\GlobalFungi_5_sample_metadata.csv"
output_file = r"C:\Users\holly\Documents\projects\fungi_database\joined_fungi_metadata.csv"

# Load metadata (also check if tab-delimited, add sep='\t' if so)
metadata = pd.read_csv(fungi_metadata, sep='\t')  # try with or without sep depending on metadata file

header_written = False
chunk_size = 500  # smaller chunk due to many columns

for chunk in pd.read_csv(fungi, chunksize=chunk_size, sep='\t', engine='python'):
    merged = pd.merge(chunk, metadata, on='sample_ID', how='left')
    merged.to_csv(output_file, mode='a', index=False, header=not header_written)
    header_written = True

print("Join complete. Output saved to", output_file)
