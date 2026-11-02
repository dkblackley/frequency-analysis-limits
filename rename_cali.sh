#!/bin/bash

# Define the target directory and the strings
TARGET_DIR="databases"
SEARCH_STRING="cali_50_"
REPLACE_STRING="cali_"

# Verify the databases directory exists before proceeding
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' not found in the current path."
  exit 1
fi

echo "Scanning '$TARGET_DIR' for files and directories containing '$SEARCH_STRING'..."

# Find all matching files and directories
# -depth: Processes directory contents before the directory itself (bottom-up)
# -name: Looks for the exact string anywhere in the filename
# -execdir: Executes the bash command from the directory containing the matched file
find "$TARGET_DIR" -depth -name "*${SEARCH_STRING}*" -execdir bash -c '
  search=$1
  replace=$2
  shift 2
  
  for item; do
    # Only process the base name (handles the "./" prefix injected by find)
    base_item="${item#./}"
    
    # Generate the new name by substituting the search string with the replace string
    new_name="${base_item//$search/$replace}"
    
    # Safety check: ensure we do not overwrite an existing file/folder
    if [ -e "$new_name" ]; then
      echo "⚠️  Skipping: Cannot rename \"$base_item\" to \"$new_name\" (Destination already exists)."
    else
      mv -- "$base_item" "$new_name"
      echo "✅ Renamed: \"$base_item\" -> \"$new_name\""
    fi
  done
' _ "$SEARCH_STRING" "$REPLACE_STRING" {} +

echo "Operation complete."
