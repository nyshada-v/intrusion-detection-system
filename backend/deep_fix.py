import os

file_path = "capture.py"

try:
    # Read the file as binary first
    with open(file_path, "rb") as f:
        blob = f.read()

    # Remove the actual null bytes
    clean_blob = blob.replace(b"\x00", b"")

    # Try to decode it as UTF-8, ignoring errors, to get clean text
    # This strips away non-textual corruption
    text_content = clean_blob.decode("utf-8", errors="ignore")

    # Write it back as a fresh, clean UTF-8 text file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"✨ Deep Clean complete. {file_path} is now a standard UTF-8 text file.")

except Exception as e:
    print(f"❌ Failed to clean: {e}")