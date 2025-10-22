import os
import sys
import json

from glob import glob
from shutil import move

import nipype.pipeline.engine as pe
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import BET

from bidsonym.reports import setup_logging

# Add any other missing imports if needed


def check_outpath(bids_dir, subject_label):
    """
    Check if output paths exist, if not create them.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to be checked (without 'sub-').
    """

    # Construct the output path for the subject's processed data
    # Following BIDS structure: sourcedata/bidsonym/sub-{subject_label}
    # The subject_label should not include the 'sub-' prefix as it's added here
    out_path = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s" % subject_label)

    # Check if the directory doesn't exist
    if not os.path.isdir(out_path):
        # Create the directory (and any necessary parent directories)
        # makedirs() will create intermediate directories if they don't exist
        os.makedirs(out_path)


def copy_no_deid(bids_dir, subject_label, image_file, session=None):
    """
    Move original non-defaced images to sourcedata.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to move (without 'sub-').
    image_file : str
        Original non-defaced image.
    session : str
        Session label (if applicable).

    Returns
    -------
    moved_img_path : str
        Path to moved original non-defaced image.
    """

    # Construct the destination path based on whether session data is provided
    if session is not None:
        path = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/ses-%s" % (subject_label, session))
    else:
        path = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s" % subject_label)
    
    # Extract just the filename from the full image file path
    outfile = image_file[image_file.rfind('/') + 1:]
    moved_img_path = os.path.join(path, outfile)

    # Check if this specific file already exists (not just the directory)
    if os.path.exists(moved_img_path):
        print(f"Warning: Non-de-identified file {moved_img_path} already exists. Skipping copy.")
        return moved_img_path
    
    # Create the directory structure if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Copy the file to preserve the original
    import shutil
    shutil.copy2(image_file, moved_img_path)
    
    print(f"Copied original file to: {moved_img_path}")
    return moved_img_path


def check_meta_data(bids_dir, subject_label, prob_fields=None, session=None, modalities=None):
    """
    Extract meta-data from image headers and json files and
    subsequently evaluate values based on default keys or
    user specified keys. Outputs are csv files containing
    DataFrames with keys, values and markers concerning values.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to be checked (without 'sub-').
    prob_fields : list, optional
        List of meta-data keys ('str') that should be evaluated.
    session : str, optional
        Session label (if applicable, without 'ses-').
    modalities : list, optional
        List of modalities to process (e.g., ['T1w', 'T2w', 'FLAIR']).
    """
    
    import os
    import json
    import pandas as pd
    from glob import glob
    from bids import BIDSLayout
    
    # Initialize BIDS layout for structured querying
    layout = BIDSLayout(bids_dir)
    
    # Default modalities if none specified
    if modalities is None:
        modalities = ['T1w']
    
    # Default problematic fields to check if none specified
    if prob_fields is None:
        prob_fields = [
            'AcquisitionDate', 'AcquisitionTime', 'InstitutionName', 
            'InstitutionAddress', 'StationName', 'ManufacturerModelName',
            'DeviceSerialNumber', 'SoftwareVersions', 'StudyDate',
            'StudyTime', 'SeriesDate', 'SeriesTime', 'StudyID',
            'StudyInstanceUID', 'SeriesInstanceUID', 'StudyDescription',
            'SeriesDescription', 'PatientName', 'PatientID', 'PatientBirthDate',
            'PatientSex', 'PatientAge', 'PatientWeight', 'PatientPosition'
        ]
    
    print(f"\nChecking metadata for subject {subject_label}")
    if session:
        print(f"Session: {session}")
    print(f"Modalities: {modalities}")
    print(f"Checking for potentially identifying fields: {prob_fields}")
    
    # Find NIfTI image files and JSON metadata files for the specified criteria
    list_subject_image_files = []
    list_sub_meta_files = []
    
    for modality in modalities:
        # Query BIDS layout for specific subject, session, and modality
        if session is not None:
            # Get images for specific session and modality
            images = layout.get(
                subject=subject_label,
                session=session,
                suffix=modality,
                extension='nii.gz',
                return_type='filename'
            )
            # Get JSON files for specific session and modality
            json_files = layout.get(
                subject=subject_label,
                session=session,
                suffix=modality,
                extension='json',
                return_type='filename'
            )
        else:
            # Get images for all sessions of this subject and modality
            images = layout.get(
                subject=subject_label,
                suffix=modality,
                extension='nii.gz',
                return_type='filename'
            )
            # Get JSON files for all sessions of this subject and modality
            json_files = layout.get(
                subject=subject_label,
                suffix=modality,
                extension='json',
                return_type='filename'
            )
        
        list_subject_image_files.extend(images)
        list_sub_meta_files.extend(json_files)
    
    # Find dataset-level JSON metadata files (at root of BIDS directory)
    list_task_meta_files = []
    for modality in modalities:
        if 'func' in modality or 'bold' in modality:
            # For functional data, include task JSON files
            task_files = glob(os.path.join(bids_dir, 'task-*_bold.json'))
            list_task_meta_files.extend(task_files)
        else:
            # For anatomical data, check for modality-specific dataset files
            dataset_files = glob(os.path.join(bids_dir, f'*_{modality}.json'))
            list_task_meta_files.extend(dataset_files)
    
    # Combine dataset-level and subject/session-specific JSON files
    list_meta_files = list_task_meta_files + list_sub_meta_files
    
    # Remove duplicates while preserving order
    list_meta_files = list(dict.fromkeys(list_meta_files))
    
    # Only proceed if we found relevant files
    if not list_subject_image_files:
        print(f'No {modalities} images found for subject {subject_label}')
        if session:
            print(f'in session {session}')
        return
    
    if not list_meta_files:
        print(f'No JSON metadata files found for subject {subject_label}')
        if session:
            print(f'in session {session}')
        print(f'and modalities {modalities}')
        return
    
    # Inform user about which files will be processed
    print(f'\nFound {len(list_subject_image_files)} image files for processing:')
    for img_file in list_subject_image_files:
        print(f'  {os.path.basename(img_file)}')
    
    print(f'\nThe following {len(list_meta_files)} metadata files will be checked:')
    for meta_file in list_meta_files:
        print(f'  {os.path.basename(meta_file)}')

    # Initialize results storage
    metadata_results = []
    
    # Process each JSON metadata file
    print(f'\nProcessing JSON metadata files...')
    for meta_file in list_meta_files:
        print(f'\nChecking: {os.path.basename(meta_file)}')
        
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Check each problematic field
            for field in prob_fields:
                if field in metadata:
                    value = metadata[field]
                    result = {
                        'file': os.path.basename(meta_file),
                        'field': field,
                        'value': str(value),
                        'potentially_identifying': True
                    }
                    metadata_results.append(result)
                    print(f'  WARNING: Found potentially identifying field: {field} = {value}')
            
            # Check for any other fields that might be identifying
            identifying_keywords = ['patient', 'name', 'id', 'date', 'time', 'institution', 'address']
            for key, value in metadata.items():
                if key not in prob_fields:
                    key_lower = key.lower()
                    if any(keyword in key_lower for keyword in identifying_keywords):
                        result = {
                            'file': os.path.basename(meta_file),
                            'field': key,
                            'value': str(value),
                            'potentially_identifying': True
                        }
                        metadata_results.append(result)
                        print(f'  WARNING: Found additional potentially identifying field: {key} = {value}')
        
        except Exception as e:
            print(f'  ERROR: Error reading {meta_file}: {e}')
    
    # Process each image file's header metadata
    print('\nProcessing image header metadata...')
    for subject_image_file in list_subject_image_files:
        print(f'\nChecking headers: {os.path.basename(subject_image_file)}')
        
        try:
            from nibabel import load
            img = load(subject_image_file)
            header = img.header
            
            # Check NIfTI header fields
            if hasattr(header, 'get_data_dtype'):
                # Check description field
                if hasattr(header, 'get_descrip'):
                    descrip = header.get_descrip()
                    if descrip and descrip.strip():
                        result = {
                            'file': os.path.basename(subject_image_file),
                            'field': 'descrip',
                            'value': str(descrip),
                            'potentially_identifying': True
                        }
                        metadata_results.append(result)
                        print(f'  WARNING: Found description in header: {descrip}')
            
            # Check for DICOM fields in NIfTI extensions
            if hasattr(img, 'get_header') and hasattr(img.get_header(), 'extensions'):
                extensions = img.get_header().extensions
                for ext in extensions:
                    if hasattr(ext, 'get_content'):
                        content = str(ext.get_content())
                        for field in prob_fields:
                            if field in content:
                                result = {
                                    'file': os.path.basename(subject_image_file),
                                    'field': f'extension_{field}',
                                    'value': 'Found in NIfTI extension',
                                    'potentially_identifying': True
                                }
                                metadata_results.append(result)
                                print(f'  WARNING: Found {field} in NIfTI extension')
        
        except Exception as e:
            print(f'  ERROR: Error reading headers from {subject_image_file}: {e}')
    
    # Generate summary report
    print(f'\n{"="*60}')
    print(f'METADATA CHECK SUMMARY')
    print(f'{"="*60}')
    
    if metadata_results:
        print(f'Found {len(metadata_results)} potentially identifying metadata fields:')
        
        # Create DataFrame for better organization
        df = pd.DataFrame(metadata_results)
        
        # Group by file for cleaner output
        for filename in df['file'].unique():
            file_results = df[df['file'] == filename]
            print(f'\nFile: {filename}:')
            for _, row in file_results.iterrows():
                print(f'   {row["field"]}: {row["value"]}')
        
        # Save results to CSV in the proper meta_data_info directory
        if session:
            output_dir = os.path.join(bids_dir, 'sourcedata', 'bidsonym', f'sub-{subject_label}', f'ses-{session}', 'meta_data_info')
            csv_filename = f'sub-{subject_label}_ses-{session}_metadata-check.csv'
        else:
            output_dir = os.path.join(bids_dir, 'sourcedata', 'bidsonym', f'sub-{subject_label}', 'meta_data_info')
            csv_filename = f'sub-{subject_label}_metadata-check.csv'
        
        # Ensure the meta_data_info directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f'\nResults saved to: {csv_path}')
        
        print(f'\nWARNING: Found potentially identifying information!')
        print(f'   Please review the results and consider removing or anonymizing')
        print(f'   the identified fields before sharing this data.')
        
    else:
        print(f'SUCCESS: No potentially identifying metadata fields found.')
        print(f'   The checked files appear to be properly de-identified.')
    
    print(f'{"="*60}')


def del_meta_data(bids_dir, subject_label, fields_del):
    """
    Delete values from specified keys in meta-data json files.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to operate on (without 'sub-').
    fields_del : list
        List of meta-data keys ('str') which value should be removed.
    """

    # Define paths for storing backed-up metadata files
    path_task_meta = os.path.join(bids_dir, "sourcedata/bidsonym/")
    path_sub_meta = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s" % subject_label)
    
    # Find all JSON metadata files at task level and subject level
    list_task_meta_files = glob(os.path.join(bids_dir, '*json'))
    list_sub_meta_files = glob(os.path.join(bids_dir, 'sub-' + subject_label, '**/*.json'), recursive=True)

    # Combine both lists for comprehensive processing
    list_meta_files = list_task_meta_files + list_sub_meta_files

    # Move original task-level JSON files to backup location in sourcedata
    # This preserves the original metadata before de-identification
    for task_meta_data_file in list_task_meta_files:
        # Extract just the filename from the full path
        task_out = task_meta_data_file[task_meta_data_file.rfind('/') + 1:]
        # Move original file to backup location
        move(task_meta_data_file, os.path.join(path_task_meta, task_out))
    
    # Move original subject-level JSON files to backup location in sourcedata
    for sub_meta_data_file in list_sub_meta_files:
        # Extract just the filename from the full path
        sub_out = sub_meta_data_file[sub_meta_data_file.rfind('/') + 1:]
        # Move original file to backup location
        move(sub_meta_data_file, os.path.join(path_sub_meta, sub_out))

    # Find the backed-up JSON files in their new locations for processing
    list_task_meta_files_deid = glob(os.path.join(bids_dir, "sourcedata/bidsonym/", '*json'))
    list_sub_meta_files_deid = glob(os.path.join(bids_dir, "sourcedata/bidsonym/",
                                                 'sub-' + subject_label, '**/*.json'),
                                    recursive=True)
    
    # Combine backed-up files for de-identification processing
    list_meta_files_deid = list_task_meta_files_deid + list_sub_meta_files_deid

    # Store the fields to delete (redundant assignment, but kept for clarity)
    fields_del = fields_del

    # Print progress information for user
    print('working on %s' % subject_label)
    print('found the following meta-data files:')
    print(*list_meta_files, sep='\n')
    print('the following fields will be deleted:')
    print(*fields_del, sep='\n')

    # Sort both lists to ensure consistent pairing of original and backup files
    list_meta_files.sort()
    list_meta_files_deid.sort()

    # Process each pair of backed-up and original files
    for meta_file_deid, meta_file in zip(list_meta_files_deid, list_meta_files):
        # Load the backed-up JSON file for processing
        with open(meta_file_deid, 'r') as json_file:
            meta_data = json.load(json_file)
            
            # Process each field marked for deletion
            for field in fields_del:
                if field in meta_data:
                    # Replace the field value with a deletion marker instead of removing the key
                    # This maintains the JSON structure while indicating the field was anonymized
                    meta_data[field] = 'deleted_by_bidsonym'
                else:
                    # Inform user if a specified field doesn't exist in this file
                    print("The field you indicated to delete does not exist in %s" % meta_file_deid)
                    continue
        
        # Write the de-identified metadata back to the original file location
        # This replaces the original file with the anonymized version
        with open(meta_file, 'w') as json_output_file:
            print('writing %s' % meta_file)
            # Use indent=4 for human-readable formatting
            json.dump(meta_data, json_output_file, indent=4)


def rename_non_deid(bids_dir, subject_label):
    """
    Rename non-deidentified files to include descriptive labels.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject (without 'sub-').
    """
    
    import os
    from glob import glob
    
    # Define the base sourcedata path for this subject
    sourcedata_base = os.path.join(bids_dir, "sourcedata", "bidsonym", f"sub-{subject_label}")
    
    if not os.path.exists(sourcedata_base):
        print(f"No sourcedata found for subject {subject_label}")
        return
    
    # Find all NIfTI and JSON files recursively in the subject's sourcedata directory
    # This will catch files in both session directories and subject root
    nifti_files = glob(os.path.join(sourcedata_base, "**", "*.nii.gz"), recursive=True)
    json_files = glob(os.path.join(sourcedata_base, "**", "*.json"), recursive=True)
    
    all_files = nifti_files + json_files
    
    print(f"Found {len(all_files)} files to rename for subject {subject_label}")
    
    # Rename each file to include the _desc-nondeid identifier
    for file_path in all_files:
        # Skip files that already have the descriptor
        if '_desc-nondeid' in file_path:
            continue
            
        # Get the directory and filename
        file_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Insert _desc-nondeid before the file extension
        if filename.endswith('.nii.gz'):
            new_filename = filename.replace('.nii.gz', '_desc-nondeid.nii.gz')
        elif filename.endswith('.json'):
            new_filename = filename.replace('.json', '_desc-nondeid.json')
        else:
            continue  # Skip files with unexpected extensions
        
        new_file_path = os.path.join(file_dir, new_filename)
        
        try:
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Error renaming {filename}: {e}")


def brain_extraction_nb(image, subject_label, bids_dir):
    """
    Setup nobrainer brainextraction command.

    Parameters
    ----------
    image : str
        Path to image that should be defaced.
    outfile : str
        Name of the defaced file.
    bids_dir : str
        Path to BIDS root directory.
    """

    import os
    from subprocess import check_call

    # Construct the output path for the brain mask
    # The mask will be saved in the subject's backup directory with descriptive naming
    # Extract the base filename and add brain mask identifier and non-deid descriptor
    outfile = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s" % subject_label,
                           image[image.rfind('/') + 1:image.rfind('.nii')] + '_brainmask_desc-nondeid.nii.gz')

    # Construct the nobrainer command for brain extraction
    # nobrainer is a deep learning-based neuroimaging tool for brain extraction
    cmd = ['nobrainer',                    # Base command
           'predict',                      # Use prediction mode
           '--model=/opt/nobrainer/models/brain-extraction-unet-128iso-model.h5',  # Pre-trained U-Net model for brain extraction
           '--verbose',                    # Enable verbose output for debugging/monitoring
           image,                          # Input image file path
           outfile,                        # Output brain mask file path
           ]
    
    # Execute the nobrainer brain extraction command
    # check_call will raise an exception if the command fails (non-zero exit code)
    # This ensures the function fails fast if brain extraction doesn't work
    check_call(cmd)


def run_brain_extraction_nb(image, subject_label, bids_dir):
    """
    Setup and run nobrainer brainextraction workflow.

    Parameters
    ----------
    image : str
        Path to image that should be defaced.
    outfile : str
        Name of the defaced file.
    bids_dir : str
        Path to BIDS root directory.
    """

    # Create a Nipype workflow for brain extraction
    brainextraction_wf = pe.Workflow('brainextraction_wf')
    
    # Create an input node to handle input data
    # IdentityInterface passes data through without modification
    inputnode = pe.Node(niu.IdentityInterface(['in_file']),
                        name='inputnode')
    
    # Create a processing node that wraps the brain_extraction_nb function
    brainextraction = pe.Node(Function(input_names=['image', 'subject_label', 'bids_dir'],
                                       output_names=['outfile'],
                                       function=brain_extraction_nb),
                              name='brainextraction')
    
    # Connect the input node to the brain extraction node
    brainextraction_wf.connect([(inputnode, brainextraction, [('in_file', 'image')])])
    
    # Set the input data - the path to the image file to be processed
    inputnode.inputs.in_file = image
    
    # Set the subject label for the brain extraction node
    # This is used to construct proper output paths and filenames
    brainextraction.inputs.subject_label = subject_label
    
    # Set the BIDS directory path for the brain extraction node
    # This defines where output files should be stored
    brainextraction.inputs.bids_dir = bids_dir
    
    # Execute the workflow
    # This runs the entire pipeline: input -> brain extraction -> output
    brainextraction_wf.run()


def run_brain_extraction_bet(image, frac, subject_label, bids_dir):
    """
    Setup and FSLs brainextraction (BET) workflow.

    Parameters
    ----------
    image : str
        Path to image that should be defaced.
    frac : float
        Fractional intensity threshold (0 - 1).
    outfile : str
        Name of the defaced file.
    bids_dir : str
        Path to BIDS root directory.
    """

    import os

    # Construct the output path for the brain-extracted image
    # The output will be saved in the subject's backup directory with descriptive naming
    # Extract the base filename and add brain mask identifier and non-deid descriptor
    outfile = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s" % subject_label,
                           image[image.rfind('/') + 1:image.rfind('.nii')] + '_brainmask_desc-nondeid.nii.gz')

    # Create a Nipype workflow for FSL BET brain extraction
    # BET (Brain Extraction Tool) is FSL's classic brain extraction algorithm
    brainextraction_wf = pe.Workflow('brainextraction_wf')
    
    # Create an input node to handle input data
    # IdentityInterface passes data through without modification
    # This serves as the entry point for data into the workflow
    inputnode = pe.Node(niu.IdentityInterface(['in_file']),
                        name='inputnode')
    
    # Create a BET (Brain Extraction Tool) node from FSL
    # mask=False means we want the brain-extracted image, not just a binary mask
    # BET uses intensity-based thresholding and morphological operations for brain extraction
    bet = pe.Node(BET(mask=False), name='bet')
    
    # Connect the input node to the BET node
    # This creates a data flow: inputnode.in_file -> bet.in_file
    brainextraction_wf.connect([
        (inputnode, bet, [('in_file', 'in_file')])
    ])
    
    # Set the input data - the path to the image file to be processed
    inputnode.inputs.in_file = image
    
    # Set the fractional intensity threshold for BET
    # This parameter controls how aggressively BET removes non-brain tissue
    # Lower values (e.g., 0.1) = more conservative, higher values (e.g., 0.7) = more aggressive
    bet.inputs.frac = float(frac)
    
    # Set the output file path for the brain-extracted image
    bet.inputs.out_file = outfile
    
    # Execute the workflow
    # This runs the entire pipeline: input -> BET brain extraction -> output
    brainextraction_wf.run()


def validate_input_dir(exec_env, bids_dir, participant_label):
    """
    Validate BIDS directory and structure via the BIDS-validator.
    Functionality copied from fmriprep.

    Parameters
    ----------
    exec_env : str
        Environment BIDSonym is run in.
    bids_dir : str
        Path to BIDS root directory.
    participant_label: str
        Label(s) of subject to be checked (without 'sub-').
    """

    import tempfile
    import subprocess
    
    # Configure the BIDS validator with a comprehensive list of warnings/errors to ignore
    # This allows validation to focus on structural issues while ignoring common
    # non-critical warnings that don't affect the defacing/anonymization process
    validator_config_dict = {
        "ignore": [
            "EVENTS_COLUMN_ONSET",           # Missing onset column in events files
            "EVENTS_COLUMN_DURATION",       # Missing duration column in events files
            "TSV_EQUAL_ROWS",               # Unequal number of rows in TSV files
            "TSV_EMPTY_CELL",               # Empty cells in TSV files
            "TSV_IMPROPER_NA",              # Improper N/A values in TSV files
            "VOLUME_COUNT_MISMATCH",        # Mismatch in volume counts between files
            "BVAL_MULTIPLE_ROWS",           # Multiple rows in bval files
            "BVEC_NUMBER_ROWS",             # Incorrect number of rows in bvec files
            "DWI_MISSING_BVAL",             # Missing bval files for DWI data
            "INCONSISTENT_SUBJECTS",        # Inconsistent subject information
            "INCONSISTENT_PARAMETERS",      # Inconsistent acquisition parameters
            "BVEC_ROW_LENGTH",              # Incorrect bvec row length
            "B_FILE",                       # Issues with b-files
            "PARTICIPANT_ID_COLUMN",        # Missing participant_id column
            "PARTICIPANT_ID_MISMATCH",      # Mismatch in participant IDs
            "TASK_NAME_MUST_DEFINE",        # Undefined task names
            "PHENOTYPE_SUBJECTS_MISSING",   # Missing subjects in phenotype files
            "STIMULUS_FILE_MISSING",        # Missing stimulus files
            "DWI_MISSING_BVEC",             # Missing bvec files for DWI data
            "EVENTS_TSV_MISSING",           # Missing events TSV files
            "TSV_IMPROPER_NA",              # Duplicate entry (intentional)
            "ACQTIME_FMT",                  # Acquisition time format issues
            "Participants age 89 or higher",  # Age-related warnings (privacy)
            "DATASET_DESCRIPTION_JSON_MISSING",  # Missing dataset description
            "FILENAME_COLUMN",              # Issues with filename columns
            "WRONG_NEW_LINE",               # Wrong newline characters
            "MISSING_TSV_COLUMN_CHANNELS",  # Missing channels column in TSV
            "MISSING_TSV_COLUMN_IEEG_CHANNELS",  # Missing iEEG channels column
            "MISSING_TSV_COLUMN_IEEG_ELECTRODES",  # Missing iEEG electrodes column
            "UNUSED_STIMULUS",              # Unused stimulus files
            "CHANNELS_COLUMN_SFREQ",        # Missing sampling frequency column
            "CHANNELS_COLUMN_LOWCUT",       # Missing low-cut filter column
            "CHANNELS_COLUMN_HIGHCUT",      # Missing high-cut filter column
            "CHANNELS_COLUMN_NOTCH",        # Missing notch filter column
            "CUSTOM_COLUMN_WITHOUT_DESCRIPTION",  # Custom columns without description
            "ACQTIME_FMT",                  # Duplicate entry (intentional)
            "SUSPICIOUSLY_LONG_EVENT_DESIGN",  # Unusually long event designs
            "SUSPICIOUSLY_SHORT_EVENT_DESIGN",  # Unusually short event designs
            "MALFORMED_BVEC",               # Malformed bvec files
            "MALFORMED_BVAL",               # Malformed bval files
            "MISSING_TSV_COLUMN_EEG_ELECTRODES",  # Missing EEG electrodes column
            "MISSING_SESSION"               # Missing session information
        ],
        "error": ["NO_T1W"],  # Still treat missing T1w images as errors (critical for defacing)
        "ignoredFiles": ['/dataset_description.json', '/participants.tsv']  # Skip these files
    }
    
    # Validate participant labels and limit validation to requested participants only
    if participant_label:
        # Get all subject directories in the BIDS dataset
        all_subs = set([s.name[4:] for s in bids_dir.glob('sub-*')])
        
        # Parse requested participant labels, handling both 'sub-' prefixed and plain labels
        selected_subs = set([s[4:] if s.startswith('sub-') else s
                             for s in participant_label])
        
        # Check for invalid participant labels (requested but not found in dataset)
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            # Create detailed error message with environment-specific troubleshooting
            error_msg = 'Data for requested participant(s) label(s) not found. Could ' \
                        'not find data for participant(s): %s. Please verify the requested ' \
                        'participant labels.'
            
            # Add Docker-specific troubleshooting information
            if exec_env == 'docker':
                error_msg += ' This error can be caused by the input data not being ' \
                             'accessible inside the docker container. Please make sure all ' \
                             'volumes are mounted properly (see https://docs.docker.com/' \
                             'engine/reference/commandline/run/#mount-volume--v---read-only)'
            
            # Add Singularity-specific troubleshooting information
            if exec_env == 'singularity':
                error_msg += ' This error can be caused by the input data not being ' \
                             'accessible inside the singularity container. Please make sure ' \
                             'all paths are mapped properly (see https://www.sylabs.io/' \
                             'guides/3.0/user-guide/bind_paths_and_mounts.html)'
            
            # Raise error with the list of problematic participant labels
            raise RuntimeError(error_msg % ','.join(bad_labels))

        # For participants not selected, add them to ignored files list
        # This optimizes validation by skipping unnecessary subjects
        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                # Use wildcard pattern to ignore entire subject directories
                validator_config_dict["ignoredFiles"].append("/sub-%s/**" % sub)
    
    # Run BIDS validation using temporary configuration file
    with tempfile.NamedTemporaryFile('w+') as temp:
        # Write the validator configuration to a temporary JSON file
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        
        try:
            # Execute BIDS validator with custom configuration
            # -c flag specifies the path to the configuration file
            subprocess.check_call(['bids-validator', bids_dir, '-c', temp.name])
        except FileNotFoundError:
            # Handle case where BIDS validator is not installed
            # Print to stderr to distinguish from normal output
            print("bids-validator does not appear to be installed", file=sys.stderr)


def deface_image(image, warped_mask, outfile):
    """
    Deface other contrast/modality image using the 
    defaced T1w image as deface mask.

    Parameters
    ----------
    image : str
        Path to image.
    warped_mask : str
        Path to warped defaced T1w image.
    outfile: str
        Name of the defaced file.
    """
    
    # Import all required modules within the function
    # This is necessary when the function is used as a Nipype Function node
    import numpy as np
    from nibabel import load, Nifti1Image
    from nilearn.image import math_img

    # Load the input image and the warped mask image
    infile_img = load(image)
    warped_mask_img = load(warped_mask)
    
    # Convert the warped mask to binary (0 or 1 values)
    # Any positive values in the mask become 1, others become 0
    warped_mask_img = math_img('img > 0', img=warped_mask_img)
    
    try:
        # Attempt to apply the mask by element-wise multiplication
        # This removes facial regions by setting them to 0
        outdata = infile_img.get_fdata().squeeze() * warped_mask_img.get_fdata()
    except ValueError:
        # Handle cases where image dimensions don't match
        # This typically happens with multi-volume/4D images

        # Create a stack of mask data to match the last dimension of the input image
        # This replicates the 3D mask across all volumes in a 4D image
        tmpdata = np.stack([warped_mask_img.get_fdata()] *
                           infile_img.get_fdata().shape[-1], axis=-1)
        
        # Apply the replicated mask to all volumes
        outdata = infile_img.get_fdata() * tmpdata

    # Create a new NIfTI image with the defaced data
    # Preserve the original image's spatial transformation (affine) and metadata (header)
    masked_brain = Nifti1Image(outdata, infile_img.affine,
                               infile_img.header)
    
    # Save the defaced image to the specified output file
    masked_brain.to_filename(outfile)


def clean_up_files(bids_dir, subject_label, session=None):
    """
    Restructure BIDSonym outcomes following BIDS conventions.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to move (without 'sub-').
    session : str, optional
        If multiple sessions exist, create session specific
        structure.
    """
    
    # Add missing import
    import os
    from glob import glob
    from shutil import move

    # Create output paths based on whether session information is provided
    if session is not None:
        # For multi-session datasets, organize files by session
        out_path_anat = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/ses-%s/anat"
                                     % (subject_label, session))
        out_path_qc = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/ses-%s/QC"
                                   % (subject_label, session))
        out_path_info = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/ses-%s/meta_data_info"
                                     % (subject_label, session))
        
        # Find all files for this specific subject and session combination
        # Look for NIfTI image files (original, defaced, brain masks, etc.)
        list_imaging_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                               'sub-' + subject_label + '_ses-' + session + '*.nii.gz'))
        
        # Find brain mask files specifically (these are anatomical derivatives)
        list_brainmask_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                                 'sub-' + subject_label + '_ses-' + session + '*_brainmask*.nii.gz'))
        
        # Find JSON files that correspond to the NIfTI images (should go with images in anat/)
        list_image_json_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                                  'sub-' + subject_label + '_ses-' + session + '*_desc-nondeid.json'))
        
        # Find visualization files (PNG images showing before/after defacing) - go to QC/
        list_graphics = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                          'sub-' + subject_label + '_ses-' + session + '*.png'))
        
        # Find animated GIF files (showing defacing process or comparisons) - go to QC/
        list_gifs = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                      'sub-' + subject_label + '_ses-' + session + '*.gif'))
        
        # Find metadata analysis CSV files (from check_meta_data function)
        list_info_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                            'sub-' + subject_label + '_ses-' + session + '*.csv'))
        
        # Find other JSON metadata files that don't correspond to images (task-level, etc.)
        all_json_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label,
                                           'sub-' + subject_label + '_ses-' + session + '*.json'))
        list_other_json_files = [f for f in all_json_files if f not in list_image_json_files]
        
    else:
        # For single-session datasets, organize files without session subdirectories
        out_path_anat = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/anat" % subject_label)
        out_path_qc = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/QC" % subject_label)
        out_path_info = os.path.join(bids_dir, "sourcedata/bidsonym/sub-%s/meta_data_info" % subject_label)
        
        # Find all files for this subject (no session filtering)
        # Look for NIfTI image files in the subject's directory
        list_imaging_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*.nii.gz'))
        
        # Find brain mask files specifically (these are anatomical derivatives)
        list_brainmask_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*_brainmask*.nii.gz'))
        
        # Find JSON files that correspond to the NIfTI images (should go with images in anat/)
        list_image_json_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*_desc-nondeid.json'))
        
        # Find visualization PNG files - go to QC/
        list_graphics = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*.png'))
        
        # Find animated GIF files - go to QC/
        list_gifs = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*.gif'))
        
        # Find metadata analysis CSV files
        list_info_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*.csv'))
        
        # Find other JSON metadata files that don't correspond to images
        all_json_files = glob(os.path.join(bids_dir, 'sourcedata/bidsonym/sub-%s' % subject_label, '*.json'))
        list_other_json_files = [f for f in all_json_files if f not in list_image_json_files]

    # Create output directories if they don't exist
    # Directory for anatomical files (NIfTI images + brain masks + their corresponding JSON files)
    if not os.path.isdir(out_path_anat):
        os.makedirs(out_path_anat)
    
    # Directory for quality control visualizations (PNG plots and GIF animations)
    if not os.path.isdir(out_path_qc):
        os.makedirs(out_path_qc)
    
    # Directory for meta-data-related files (CSV analysis files and other JSON files)
    if not os.path.isdir(out_path_info):
        os.makedirs(out_path_info)

    # Move anatomical files (NIfTI images + brain masks + their corresponding JSON metadata) to anat directory
    anat_files = list_imaging_files + list_brainmask_files + list_image_json_files
    
    # Remove duplicates that might occur if brain masks are already in imaging_files
    anat_files = list(set(anat_files))
    
    for anat_file in anat_files:
        # Extract just the filename from the full path
        file_out = anat_file[anat_file.rfind('/') + 1:]
        # Move file to the organized anat directory
        move(anat_file, os.path.join(out_path_anat, file_out))

    # Move quality control visualization files (PNG plots + GIF animations) to QC directory
    qc_files = list_graphics + list_gifs
    for qc_file in qc_files:
        # Extract just the filename from the full path
        file_out = qc_file[qc_file.rfind('/') + 1:]
        # Move file to the organized QC directory
        move(qc_file, os.path.join(out_path_qc, file_out))

    # Move meta-data analysis files to the organized info directory
    # This includes CSV analysis files and other JSON metadata files (not image-specific)
    for info_file in list_info_files + list_other_json_files:
        # Extract just the filename from the full path
        file_out = info_file[info_file.rfind('/') + 1:]
        # Move file to the organized metadata info directory
        move(info_file, os.path.join(out_path_info, file_out))


def revert_bidsonym(bids_dir, subject_label, session=None, confirm=True):
    """
    Revert the BIDSonym process by copying back non-defaced images and
    metadata from sourcedata and removing all BIDSonym-generated files.

    This function performs a complete reversal of the BIDSonym anonymization
    process by restoring original files from the sourcedata backup and removing
    all defaced/de-identified files from the main BIDS structure.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to restore (without 'sub-' prefix).
    session : str, optional
        Session label (if applicable, without 'ses-' prefix).
        If provided, only that specific session will be reverted.
    confirm : bool, optional
        If True, ask for user confirmation before proceeding.
        Default is True for safety to prevent accidental data loss.
    
    Returns
    -------
    bool
        True if reversion was successful, False otherwise.
    """
    
    import os
    import shutil
    from glob import glob
    from shutil import copy2
    
    # Set up logging system
    log_print, log_path = setup_logging(bids_dir, subject_label, session, "bidsonymrevert")
    
    # Build session description for messages
    session_desc = f" (session: {session})" if session is not None else ""
    
    # Display header with subject and session information
    log_print("BIDSonym Revert")
    log_print(f"Subject: sub-{subject_label}")
    
    # Build descriptive message including session if provided
    if session is not None:
        log_print(f"Session: ses-{session}")
        log_print("Processing multi-session dataset structure")
    else:
        log_print("Processing single-session dataset structure")
    
    if log_path:
        log_print(f"Log file created: {log_path}")
    
    # Define paths based on whether this is a session-specific or single-session dataset
    if session is not None:
        # Multi-session dataset: target specific session directory
        subject_dir = os.path.join(bids_dir, f"sub-{subject_label}", f"ses-{session}")
        sourcedata_subject_dir = os.path.join(bids_dir, "sourcedata", "bidsonym", f"sub-{subject_label}", f"ses-{session}")
        # Base directory contains all sessions for this subject
        sourcedata_base_dir = os.path.join(bids_dir, "sourcedata", "bidsonym", f"sub-{subject_label}")
    else:
        # Single-session dataset: target subject directory directly
        subject_dir = os.path.join(bids_dir, f"sub-{subject_label}")
        sourcedata_subject_dir = os.path.join(bids_dir, "sourcedata", "bidsonym", f"sub-{subject_label}")
        # Base and subject directories are the same for single-session
        sourcedata_base_dir = sourcedata_subject_dir
    
    # Display path information for transparency
    log_print(f"BIDS directory: {bids_dir}")
    log_print(f"Target subject directory: {subject_dir}")
    log_print(f"BIDSonym sourcedata directory: {sourcedata_base_dir}")
    log_print("-" * 60)
    
    # Check if sourcedata directory exists - this indicates BIDSonym was previously run
    log_print(f"\n Checking for BIDSonym backup data{session_desc}...")
    if not os.path.exists(sourcedata_base_dir):
        log_print(f"ERROR: No BIDSonym sourcedata found for subject {subject_label}{session_desc}", "ERROR")
        log_print(f"   Expected location: {sourcedata_base_dir}", "ERROR")
        log_print("   This indicates BIDSonym was never run on this subject, or", "ERROR")
        log_print("   the backup data has been manually removed.", "ERROR")
        return False
    else:
        log_print(f"Found BIDSonym backup directory: {sourcedata_base_dir}")
    
    # Find all original files in sourcedata directory tree
    # These files have 'desc-nondeid' identifier and represent the original, non-anonymized data
    log_print(f"\nScanning for original (non-anonymized) files{session_desc}...")
    
    # Look for files with 'desc-nondeid' identifier in the main sourcedata directory
    # Use recursive search to handle both organized and unorganized file structures
    original_images = glob(os.path.join(sourcedata_base_dir, "**/*desc-nondeid.nii.gz"), recursive=True)
    original_json_files = glob(os.path.join(sourcedata_base_dir, "**/*desc-nondeid.json"), recursive=True)
    
    # Also check organized subdirectories that may exist if clean_up_files was run
    # These subdirectories separate anatomical files from metadata and QC for better organization
    anat_subdir = os.path.join(sourcedata_base_dir, "anat")  # Changed from "images" to "anat"
    qc_subdir = os.path.join(sourcedata_base_dir, "QC")
    metadata_subdir = os.path.join(sourcedata_base_dir, "meta_data_info")
    
    # Check if organized anat subdirectory exists and scan it
    if os.path.exists(anat_subdir):
        log_print(f"   Checking organized anat subdirectory: {anat_subdir}")
        additional_images = glob(os.path.join(anat_subdir, "*desc-nondeid.nii.gz"))
        additional_brainmasks = glob(os.path.join(anat_subdir, "*brainmask*.nii.gz"))
        additional_json = glob(os.path.join(anat_subdir, "*desc-nondeid.json"))
        original_images.extend(additional_images)
        original_images.extend(additional_brainmasks)  # Include brain masks with images
        original_json_files.extend(additional_json)
    
    # Check if QC subdirectory exists (contains visualization files, not for restoration)
    if os.path.exists(qc_subdir):
        log_print(f"   Found QC visualization directory: {qc_subdir}")
        qc_files = glob(os.path.join(qc_subdir, "*.png")) + glob(os.path.join(qc_subdir, "*.gif"))
        log_print(f"   QC directory contains {len(qc_files)} visualization files (will be removed)")
        
    # Check if organized metadata subdirectory exists and scan it    
    if os.path.exists(metadata_subdir):
        log_print(f"   Checking organized metadata subdirectory: {metadata_subdir}")
        additional_analysis_json = glob(os.path.join(metadata_subdir, "*.json"))
        # Only add non-desc-nondeid JSON files (these are analysis/task-level files)
        additional_analysis_json = [f for f in additional_analysis_json if 'desc-nondeid' not in f]
        original_json_files.extend(additional_analysis_json)
    
    # Find current defaced/modified files in main BIDS structure that need to be replaced
    # These are the anonymized files that will be removed and replaced with originals
    log_print(f"\nScanning current BIDS structure for files to replace{session_desc}...")
    if session is not None:
        # For session-specific reversion, only scan the target session directory
        log_print(f"   Scanning session-specific directory: {subject_dir}")
        current_images = glob(os.path.join(subject_dir, "**/*.nii.gz"), recursive=True)
        current_json_files = glob(os.path.join(subject_dir, "**/*.json"), recursive=True)
    else:
        # For single-session reversion, scan the entire subject directory
        log_print(f"   Scanning subject directory: {subject_dir}")
        current_images = glob(os.path.join(subject_dir, "**/*.nii.gz"), recursive=True)
        current_json_files = glob(os.path.join(subject_dir, "**/*.json"), recursive=True)
    
    # Validate that we found backup files to restore
    # If no original files are found, this suggests BIDSonym was never run or backup data is missing
    if not original_images and not original_json_files:
        log_print(f"  WARNING: No original files found in sourcedata for subject {subject_label}{session_desc}", "WARNING")
        log_print("   This may indicate that:", "WARNING")
        log_print("   - BIDSonym was not previously run on this subject/session", "WARNING")
        log_print("   - The backup files were manually removed", "WARNING")
        log_print("   - The BIDSonym process was incomplete or failed", "WARNING")
        log_print("   - Files may be in a different location or naming convention", "WARNING")
        return False
    
    # Display comprehensive summary of what will be restored
    log_print("\n REVERSION SUMMARY:")
    log_print("=" * 60)
    
    log_print(f" Original image files to restore: {len(original_images)}")
    if original_images:
        for img in original_images[:3]:  # Show first 3
            log_print(f"    {os.path.basename(img)}")
        if len(original_images) > 3:
            log_print(f"   ... and {len(original_images) - 3} more image files")
    
    log_print(f"\n Original JSON metadata files to restore: {len(original_json_files)}")
    if original_json_files:
        for json_file in original_json_files[:3]:  # Show first 3
            log_print(f"    {os.path.basename(json_file)}")
        if len(original_json_files) > 3:
            log_print(f"   ... and {len(original_json_files) - 3} more JSON files")
    
    log_print("\n  Current files to be removed:")
    log_print(f"   - {len(current_images)} defaced/modified image files")
    log_print(f"   - {len(current_json_files)} de-identified JSON files")
    
    log_print("\n Directories to be cleaned up:")
    log_print(f"   - {sourcedata_base_dir}")
    
    # Check if we'll remove the entire bidsonym directory structure
    # This helps inform the user about the scope of cleanup
    bidsonym_dir = os.path.join(bids_dir, "sourcedata", "bidsonym")
    remaining_subjects = []
    if os.path.exists(bidsonym_dir):
        # Find other subjects that have BIDSonym data (excluding current subject)
        remaining_subjects = [
            d for d in os.listdir(bidsonym_dir)
            if os.path.isdir(os.path.join(bidsonym_dir, d))
            and d != f"sub-{subject_label}"
        ]
    # Inform user about directory cleanup scope
    if not remaining_subjects:
        log_print(f"   - {bidsonym_dir} (no other subjects remain)")
        sourcedata_dir = os.path.join(bids_dir, "sourcedata")
        # Check if sourcedata will be completely empty after cleanup
        if os.path.exists(sourcedata_dir) and len(os.listdir(sourcedata_dir)) == 1:
            log_print(f"   - {sourcedata_dir} (will be empty)")
    else:
        log_print(f"   Note: {len(remaining_subjects)} other subjects remain in BIDSonym sourcedata")
    
    # Confirmation prompt with clear warning
    if confirm:
        log_print("\n" + "=" * 60)
        log_print("  IMPORTANT WARNING:")
        log_print("   This will permanently replace all defaced/de-identified files")
        log_print(f"   with the original non-anonymized versions for subject {subject_label}{session_desc}.")
        log_print("   This action cannot be undone!")
        if session is not None:
            log_print(f"   Only session '{session}' will be reverted for this subject.")
        else:
            log_print("   All sessions/data for this subject will be reverted.")
        log_print("=" * 60)
        
        # Note: We still need to use regular input() for user interaction
        response = input("\nType 'yes' to proceed with BIDSonym reversion: ")
        log_print(f"User response to confirmation prompt: '{response}'", "INFO")
        
        if response.lower() != 'yes':
            log_print(" BIDSonym reversion cancelled by user.", "INFO")
            return False
    
    try:
        # Step 1: Remove current defaced/modified files from main BIDS structure
        log_print(f"\n  STEP 1: Removing defaced/de-identified files{session_desc}...")
        log_print(f"   Cleaning up main BIDS directory: {subject_dir}")
        
        # Initialize counters to track removal progress
        removed_images = 0
        removed_json = 0
        
        # Remove all current image files (these are defaced/anonymized versions)
        for img_file in current_images:
            try:
                os.remove(img_file)
                removed_images += 1
                log_print(f"      Removed image: {os.path.basename(img_file)}")
            except OSError as e:
                log_print(f"       WARNING: Could not remove {os.path.basename(img_file)}: {e}", "WARNING")
        
        # Remove all current JSON files (these contain de-identified metadata)
        for json_file in current_json_files:
            try:
                os.remove(json_file)
                removed_json += 1
                log_print(f"      Removed JSON: {os.path.basename(json_file)}")
            except OSError as e:
                log_print(f"       WARNING: Could not remove {os.path.basename(json_file)}: {e}", "WARNING")
                
        log_print(f"   Summary: Removed {removed_images} images and {removed_json} JSON files")
        
        # Step 2: Restore original image files to their proper BIDS locations
        log_print(f"\n STEP 2: Restoring original image files{session_desc}...")
        log_print("   Copying from sourcedata back to main BIDS structure")
        
        # Initialize counter to track restoration progress
        restored_images = 0
        
        # Process each original image file found in sourcedata
        for original_img in original_images:
            # Remove the BIDSonym identifier to get the original BIDS filename
            original_basename = os.path.basename(original_img)
            restored_basename = original_basename.replace('_desc-nondeid', '')
            
            # Determine where this file should go in the BIDS structure
            # This logic handles files from both organized and unorganized sourcedata
            if "anat/" in original_img:  # Changed from "images/" to "anat/"
                # File is in organized structure - extract just the filename
                relative_path = restored_basename
            else:
                # File is in root of subject sourcedata directory
                relative_path = restored_basename
            
            # Determine target directory based on BIDS file naming conventions
            # Parse the filename to understand the modality and construct proper path
            if '_T1w.' in restored_basename:
                modality_dir = 'anat'
            elif '_T2w.' in restored_basename:
                modality_dir = 'anat'
            elif '_FLAIR.' in restored_basename:
                modality_dir = 'anat'
            elif '_func.' in restored_basename or '_task-' in restored_basename:
                modality_dir = 'func'
            elif '_dwi.' in restored_basename:
                modality_dir = 'dwi'
            else:
                # Default to anat for anatomical images
                modality_dir = 'anat'
            
            # Construct the full target path in the main BIDS structure
            target_path = os.path.join(subject_dir, modality_dir, restored_basename)
            
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            try:
                # Copy the original file back to its BIDS location
                copy2(original_img, target_path)
                restored_images += 1
                log_print(f"      Restored: {restored_basename}")
            except OSError as e:
                log_print(f"       ERROR: Could not restore {restored_basename}: {e}", "ERROR")
        
        log_print(f"   Summary: Restored {restored_images} image files")
        
        # Step 3: Restore original JSON metadata files
        log_print(f"\n STEP 3: Restoring original JSON metadata files{session_desc}...")
        
        # Initialize counter to track JSON restoration progress
        restored_json = 0
        
        # Process each original JSON file found in sourcedata
        for original_json in original_json_files:
            # Remove the BIDSonym identifier to get the original BIDS filename
            original_basename = os.path.basename(original_json)
            restored_basename = original_basename.replace('_desc-nondeid', '')
            
            # Determine target directory based on associated image modality
            if '_T1w.' in restored_basename or '_T2w.' in restored_basename or '_FLAIR.' in restored_basename:
                modality_dir = 'anat'
            elif '_func.' in restored_basename or '_task-' in restored_basename:
                modality_dir = 'func'
            elif '_dwi.' in restored_basename:
                modality_dir = 'dwi'
            else:
                modality_dir = 'anat'  # Default
            
            # Construct the full target path
            target_path = os.path.join(subject_dir, modality_dir, restored_basename)
            
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            try:
                # Copy the original JSON file back to its BIDS location
                copy2(original_json, target_path)
                restored_json += 1
                log_print(f"      Restored: {restored_basename}")
            except OSError as e:
                log_print(f"       ERROR: Could not restore {restored_basename}: {e}", "ERROR")
        
        log_print(f"   Summary: Restored {restored_json} JSON files")
        
        # Step 4: Clean up BIDSonym sourcedata directories
        log_print(f"\n STEP 4: Cleaning up BIDSonym sourcedata{session_desc}...")
        
        try:
            # Remove the subject-specific sourcedata directory
            if os.path.exists(sourcedata_base_dir):
                shutil.rmtree(sourcedata_base_dir)
                log_print(f"   Removed: {sourcedata_base_dir}")
            
            # Check if we should remove the entire bidsonym directory
            bidsonym_dir = os.path.join(bids_dir, "sourcedata", "bidsonym")
            if os.path.exists(bidsonym_dir) and not os.listdir(bidsonym_dir):
                # Directory is empty, remove it
                os.rmdir(bidsonym_dir)
                log_print(f"   Removed empty directory: {bidsonym_dir}")
                
                # Check if sourcedata is now empty
                sourcedata_dir = os.path.join(bids_dir, "sourcedata")
                if os.path.exists(sourcedata_dir) and not os.listdir(sourcedata_dir):
                    os.rmdir(sourcedata_dir)
                    log_print(f"   Removed empty directory: {sourcedata_dir}")
                    
        except OSError as e:
            log_print(f"   WARNING: Could not fully clean up sourcedata: {e}", "WARNING")
        
        # Final success message
        log_print(f"\n SUCCESS: BIDSonym reversion completed for subject {subject_label}{session_desc}!")
        log_print("=" * 60)
        log_print(" All original files have been restored to the main BIDS structure.")
        log_print(" All BIDSonym-generated files and directories have been removed.")
        log_print("=" * 60)
        
        return True
        
    except Exception as e:
        # Handle any unexpected errors during the reversion process
        log_print(f"\n ERROR: BIDSonym reversion failed for subject {subject_label}{session_desc}", "ERROR")
        log_print(f"   Error details: {str(e)}", "ERROR")
        log_print("   The dataset may be in an inconsistent state.", "ERROR")
        log_print("   Please check the files manually and consider restoring from backup.", "ERROR")
        return False
