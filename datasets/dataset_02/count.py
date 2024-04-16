from pathlib import Path

# Define the base path for the dataset directory
base_path = Path("")


# Function to count files in each subfolder
def count_files_in_subfolders(base_path):
    count_dict = {}
    # Iterate through each folder in the base path
    for folder in base_path.iterdir():
        if folder.is_dir():  # Make sure it is a directory
            # Get the count of files in the subfolders
            subfolder_count = sum(
                len(list(subfolder.glob('*'))) for subfolder in folder.iterdir() if subfolder.is_dir())
            # Store the count with the folder name
            count_dict[folder.name] = subfolder_count
    return count_dict


# Get the counts
folder_file_counts = count_files_in_subfolders(base_path)
print(folder_file_counts)

# can see if there is a 'test_dataset''train_dataset' 'valid_dataset' and print the count of each
