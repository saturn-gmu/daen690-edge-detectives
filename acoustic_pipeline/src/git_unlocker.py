import os
from pathlib import Path

def find_largest_file(directory="."):
    directory = Path(directory).resolve()
    largest_file = None
    largest_size = 0

    for root, _, files in os.walk(directory):
        for name in files:
            file_path = Path(root) / name
            try:
                size = file_path.stat().st_size
                if size > largest_size:
                    largest_file = file_path
                    largest_size = size
            except (OSError, PermissionError):
                continue  # Skip files we can't read

    if largest_file:
        print(f"📦 Largest file:\n{largest_file}")
        print(f"📏 Size: {largest_size / (1024 ** 2):.2f} MB")
    else:
        print("❌ No files found or accessible in this directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find the largest file in a directory.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to search (default: current)")
    args = parser.parse_args()

    find_largest_file(args.directory)
