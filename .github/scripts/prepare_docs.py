import os
import glob
import re
import shutil

folders_to_copy = ["src", "scripts"]
for folder in folders_to_copy:
    if os.path.exists(folder):
        shutil.copytree(folder, f"docs/{folder}", dirs_exist_ok=True)

for filepath in glob.glob("docs/**/*.md", recursive=True):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # RULE A: Fix links pointing OUT to src/ or scripts/
    # Logic: Because the folders were moved one level deeper, we remove exactly ONE '../'
    content = re.sub(r"\]\(\.\./((?:\.\./)*)(src|scripts)/([^)]*)\)", r"](\1\2/\3)", content)

    # RULE B: Fix links pointing FROM the copied files back TO the original docs/ folder
    # Logic: Since these files are now inside docs/, the 'docs/' segment in the path is redundant.
    content = re.sub(r"\]\(((?:\.\./)+)docs/([^)]*)\)", r"](\1\2)", content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Successfully imported external files and adjusted markdown links.")
