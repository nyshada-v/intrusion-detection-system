import os

file_path = "capture.py"

if os.path.exists(file_path):
    # Read as binary to catch the null bytes (\x00)
    with open(file_path, "rb") as f:
        data = f.read()

    # Remove null bytes
    clean_data = data.replace(b"\x00", b"")

    # Overwrite the original file with clean data
    with open(file_path, "wb") as f:
        f.write(clean_data)

    print(f"✅ Successfully scrubbed null bytes from {file_path}")
else:
    print(f"❌ Error: {file_path} not found in the current directory.")