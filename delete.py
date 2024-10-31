import os
import shutil

# Paths to the directories
csv_dir = 'outputs/csv'
graphs_dir = 'outputs/graphs'

def delete_folder_contents(folder_path):
    if os.path.exists(folder_path):
        print(f"Accessing {folder_path}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    print(f"Deleting file: {file_path}")
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    print(f"Deleting directory and contents: {file_path}")
                    shutil.rmtree(file_path)  # Remove the directory and all its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory does not exist: {folder_path}")

# Delete contents of both folders
delete_folder_contents(csv_dir)
delete_folder_contents(graphs_dir)

print("Contents of 'output/csv' and 'output/graphs' have been deleted.")
