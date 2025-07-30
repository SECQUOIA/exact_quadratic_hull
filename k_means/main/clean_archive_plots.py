import glob
import os
import shutil


def clean_archive_plots() -> None:
    """
    Delete all plot folders in the archive directory.
    """
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    archive_dir = os.path.join(data_dir, "archive")

    if not os.path.exists(archive_dir):
        print(f"Error: Archive directory not found at {archive_dir}")
        return

    # Get all subdirectories in the archive folder
    archive_folders = [d for d in glob.glob(os.path.join(archive_dir, "*")) if os.path.isdir(d)]

    if not archive_folders:
        print(f"No archive folders found in {archive_dir}")
        return

    print(f"Found {len(archive_folders)} archive folders")

    # Process each archive folder
    for archive_folder in archive_folders:
        plots_folder = os.path.join(archive_folder, "plots")
        if os.path.exists(plots_folder):
            print(f"Deleting plots folder: {plots_folder}")
            shutil.rmtree(plots_folder)
        else:
            print(f"No plots folder found in {archive_folder}")

    print("\nAll plot folders have been deleted!")


if __name__ == "__main__":
    # Ask for confirmation before proceeding
    response = input(
        "This will delete ALL plot folders in the archive directory. Continue? (y/n): "
    )
    if response.lower() == "y":
        clean_archive_plots()
    else:
        print("Operation cancelled.")
