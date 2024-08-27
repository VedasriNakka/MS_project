import shutil
import os

def folder_to_zip(folder_path, output_zip):
    """
    Convert a folder into a zip file.

    :param folder_path: Path to the folder to be zipped.
    :param output_zip: Base path and name for the output zip file.
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist")

    # Get the base name and root directory
    base_name = os.path.basename(folder_path)
    root_dir = os.path.dirname(folder_path)

    # Create a zip file
    shutil.make_archive(output_zip, 'zip', root_dir, base_name)
    print(f"Folder '{folder_path}' has been zipped into '{output_zip}.zip'")

# Example usage
folder_to_zip('/home/vedasri/Baseline_V2/results_hpo/final_experiments', '/home/vedasri/Baseline_V2/results_hpo/archive')
