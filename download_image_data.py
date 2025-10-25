#PLEASE READ: This script filters a GDC manifest file to include only .svs files and those with "DX" in their identifiers and includes instructions for downloading.

#Use the following command in the terminal to download the filtered slides:
#/Users/lucashuitema/Downloads/gdc-client download -m /Users/lucashuitema/Documents/GitHub/Immunotherapy_Response_Prediction/LUAD_LUSC_Data/Manifests/LUAD/gdc_manifest_svs_DX.txt -d /Volumes/SeagateBas/Immunotherapy/LUAD

import re

input_manifest = "LUAD_LUSC_Data/Manifests/LUAD/gdc_manifest.2025-10-25.164135.txt" # path to your manifest file
output_manifest = "LUAD_LUSC_Data/Manifests/LUAD/gdc_manifest_svs.txt" # filtered output

with open(input_manifest, "r") as infile, open(output_manifest, "w") as outfile:
    header = infile.readline() # read the header
    outfile.write(header) # keep header line

    kept = 0
    removed = 0

    for line in infile:
        if ".svs" in line:
            outfile.write(line)
            kept += 1
        else:
            removed += 1

print(f"✅ Done! Kept {kept} rows with .svs, removed {removed} others.")
print(f"Filtered manifest saved to: {output_manifest}")

input_file = output_manifest
output_file = "LUAD_LUSC_Data/Manifests/LUAD/gdc_manifest_svs_DX.txt"

pattern = re.compile(r"DX\d*", re.IGNORECASE)

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    header = infile.readline() # read the header
    outfile.write(header) # keep header line

    kept = 0
    removed = 0

    for line in infile:
        if pattern.search(line):
            kept += 1
            outfile.write(line)
        else:
            removed += 1

print(f"✅ Filtered slides written to {output_file}. Kept {kept} rows with DX, removed {removed} others.")