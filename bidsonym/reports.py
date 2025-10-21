# Import all required modules at the top
import os
from datetime import datetime

import nipype.pipeline.engine as pe
from nipype import Function
from nipype.interfaces import utility as niu


def setup_logging(bids_dir, subject_label, session=None, 
                  operation="bidsonymrevert"):
    """
    Set up logging functionality for BIDSonym operations.
    
    Creates a BIDS-compliant log file and returns a logging function that 
    writes to both console and the log file with timestamps and severity levels.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject (without 'sub-' prefix).
    session : str, optional
        Session label (without 'ses-' prefix), if applicable.
    operation : str, optional
        Name of the operation being logged (default: 'bidsonymrevert').
        
    Returns
    -------
    tuple
        (log_print_function, log_file_path)
        log_print_function: Function to print and log messages
        log_file_path: Path to the created log file
    """
    
    # Create BIDS-compliant log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    if session is not None:
        log_filename = (f"sub-{subject_label}_ses-{session}_desc-"
                        f"{operation}_{timestamp}.log")
    else:
        log_filename = (f"sub-{subject_label}_desc-{operation}_"
                        f"{timestamp}.log")
    
    # Create log directory following BIDS conventions with subject subdirectory
    log_base_dir = os.path.join(bids_dir, "sourcedata", "bidsonym", 
                                f"{operation}_logs")
    log_subject_dir = os.path.join(log_base_dir, f"sub-{subject_label}")
    os.makedirs(log_subject_dir, exist_ok=True)
    log_path = os.path.join(log_subject_dir, log_filename)
    
    # Initialize log file with header information
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("=" * 80 + '\n')
            log_file.write(f"BIDSonym {operation.title()} Log\n")
            log_file.write("=" * 80 + '\n')
            log_file.write(f"Timestamp: "
                           f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Subject: sub-{subject_label}\n")
            if session is not None:
                log_file.write(f"Session: ses-{session}\n")
            log_file.write(f"BIDS Directory: {bids_dir}\n")
            log_file.write(f"Log File: {log_path}\n")
            log_file.write("=" * 80 + '\n\n')
        
        # Create the log_print function with access to log_path
        def log_print(message="", level="INFO"):
            """Print message to console and write to log file with timestamp 
            and level."""
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp_str}] [{level}] {message}"
            
            # Print to console (original behavior)
            print(message)
            
            # Write to log file
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(log_message + '\n')
            except Exception as e:
                # If logging fails, at least show the error on console
                print(f"WARNING: Could not write to log file: {e}")
        
        return log_print, log_path
        
    except Exception as e:
        print(f"WARNING: Could not create log file {log_path}: {e}")
        
        # Fall back to regular print function if logging fails
        def log_print(message="", level="INFO"):
            print(message)
        
        return log_print, None


def plot_brainmask_overlay(bids_dir, subject_label, session=None, t2w=None):
    """
    Plot brainmask created from original non-defaced image on defaced image
    to evaluate defacing performance.
    """

    # Import all required modules within the function for Nipype compatibility
    import os
    from os.path import join as opj
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from bids import BIDSLayout
    from nilearn.plotting import find_cut_slices, plot_stat_map

    # Initialize BIDS layout to query dataset structure
    layout = BIDSLayout(bids_dir)
    
    # Define path to BIDSonym sourcedata directory for this subject
    if session is not None:
        bidsonym_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}')
    else:
        bidsonym_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}')

    # Query for T1w images based on session specification
    if session is not None:
        defaced_t1w = layout.get(
            subject=subject_label, 
            extension='nii.gz', 
            suffix='T1w',
            return_type='filename', 
            session=session
        )
    else:
        defaced_t1w = layout.get(
            subject=subject_label, 
            extension='nii.gz', 
            suffix='T1w',
            return_type='filename'
        )

    # Process each T1w image found
    for t1w in defaced_t1w:
        # Construct the EXACT path to the corresponding brain mask file
        # This ensures we get the brain mask for this specific image
        brain_mask_filename = (
            t1w[t1w.rfind('/') + 1:t1w.rfind('.nii')] + 
            '_brainmask_desc-nondeid.nii.gz'
        )
        
        # Look for the brain mask in the specific session/subject directory
        brainmask_t1w = opj(bidsonym_path, brain_mask_filename)
        
        if not os.path.exists(brainmask_t1w):
            print(f"Warning: Brain mask not found at {brainmask_t1w}")
            continue
        
        # Create figure with subplots for three orthogonal views
        fig = figure(figsize=(15, 5))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.2, hspace=0)
        
        # Generate plots for each anatomical direction
        for i, direction in enumerate(['x', 'y', 'z']):
            ax = fig.add_subplot(3, 1, i + 1)
            
            # Find optimal slice positions for this direction
            cuts = find_cut_slices(t1w, direction=direction, n_cuts=12)
            
            # Plot brain mask overlaid on defaced T1w image
            plot_stat_map(
                brainmask_t1w,           # Specific brain mask for this image
                bg_img=t1w,              # Defaced T1w as background
                display_mode=direction,   # Anatomical direction
                cut_coords=cuts,         # Slice positions
                annotate=False,          # No anatomical annotations
                dim=-1,                  # Dim background slightly
                axes=ax,                 # Use specific subplot
                colorbar=False           # No colorbar
            )
        
        # Save the plot with descriptive filename
        output_filename = (
            t1w[t1w.rfind('/') + 1:t1w.rfind('.nii')] + 
            '_desc-brainmaskdeid.png'
        )
        plt.savefig(opj(bidsonym_path, output_filename))
        plt.close()

    # Process T2w/FLAIR images if requested
    if t2w is not None:
        if session is not None:
            defaced_flair = layout.get(
                subject=subject_label, 
                extension='nii.gz', 
                suffix='FLAIR',
                return_type='filename', 
                session=session
            )
        else:
            defaced_flair = layout.get(
                subject=subject_label, 
                extension='nii.gz', 
                suffix='FLAIR',
                return_type='filename'
            )

        # Process each FLAIR image found
        for flair in defaced_flair:
            # Construct the EXACT path to the corresponding brain mask
            brain_mask_filename = (
                flair[flair.rfind('/') + 1:flair.rfind('.nii')] + 
                '_brainmask_desc-nondeid.nii.gz'
            )
            
            # Look for the brain mask in the specific session/subject directory
            brainmask_flair = opj(bidsonym_path, brain_mask_filename)
            
            if not os.path.exists(brainmask_flair):
                print(f"Warning: Brain mask not found at {brainmask_flair}")
                continue
            
            # Create figure with subplots
            fig = figure(figsize=(15, 5))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.2, hspace=0)
            
            # Generate plots for each anatomical direction
            for i, direction in enumerate(['x', 'y', 'z']):
                ax = fig.add_subplot(3, 1, i + 1)
                
                # Find optimal slice positions for FLAIR image
                cuts = find_cut_slices(flair, direction=direction, n_cuts=12)
                
                # Plot brain mask overlaid on defaced FLAIR image
                plot_stat_map(
                    brainmask_flair,         # Specific brain mask for this image
                    bg_img=flair,            # Defaced FLAIR as background
                    display_mode=direction,   # Anatomical direction
                    cut_coords=cuts,         # Slice positions
                    annotate=False,          # No anatomical annotations
                    dim=-1,                  # Dim background slightly
                    axes=ax,                 # Use specific subplot
                    colorbar=False           # No colorbar
                )
            
            # Save FLAIR plot with descriptive filename
            output_filename = (
                flair[flair.rfind('/') + 1:flair.rfind('.nii')] + 
                '_desc-brainmaskdeid.png'
            )
            plt.savefig(opj(bidsonym_path, output_filename))
            plt.close()

    # Return processed files
    return (defaced_t1w[0] if defaced_t1w else None, t2w)


def gif_defaced(bids_dir, subject_label, session=None, t2w=None):
    """
    Create animated GIFs that loop through slices of defaced images in
    orthogonal directions (x, y, z).

    This function generates animated visualizations of the defaced images
    to provide a comprehensive view of the defacing quality across all
    slices in each anatomical direction.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject to be processed (without 'sub-' prefix).
    session : str, optional
        If multiple sessions exist, create one GIF per session.
        If None, processes all sessions for the subject.
    t2w : bool, optional
        If True and T2w images exist, create GIFs for T2w images as well.

    Notes
    -----
    The generated GIFs are initially created in the subject's anatomical
    directory and then moved to the BIDSonym sourcedata directory for
    organization and storage.
    """
    
    # Import required modules within function for Nipype compatibility
    import os
    from glob import glob
    from os.path import join as opj
    from shutil import move
    from bids import BIDSLayout
    import gif_your_nifti.core as gif2nif

    # Initialize BIDS layout to query dataset structure
    layout = BIDSLayout(bids_dir)
    
    # Define path to BIDSonym sourcedata directory for this subject
    if session is not None:
        bidsonym_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}')
    else:
        bidsonym_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}')

    # Query for T1w images based on session specification
    if session is not None:
        # Get T1w images for specific session
        defaced_t1w = layout.get(
            subject=subject_label, 
            extension='nii.gz', 
            suffix='T1w',
            return_type='filename', 
            session=session
        )
        
        # Get T2w images for specific session if requested
        if t2w is not None:
            defaced_t2w = layout.get(
                subject=subject_label, 
                extension='nii.gz', 
                suffix='T2w',
                return_type='filename', 
                session=session
            )
        else:
            defaced_t2w = []
    else:
        # Get all T1w images for subject (all sessions)
        defaced_t1w = layout.get(
            subject=subject_label, 
            extension='nii.gz', 
            suffix='T1w',
            return_type='filename'
        )
        
        # Get all T2w images for subject if requested
        if t2w is not None:
            defaced_t2w = layout.get(
                subject=subject_label, 
                extension='nii.gz', 
                suffix='T2w',
                return_type='filename'
            )
        else:
            defaced_t2w = []

    # Generate GIFs for all T1w images found
    for t1_image in defaced_t1w:
        try:
            # Create animated GIF showing slices through the T1w image
            gif2nif.write_gif_normal(t1_image)
        except Exception as e:
            print(f"Warning: Could not create GIF for {t1_image}: {e}")

    # Generate GIFs for T2w images if requested
    for t2_image in defaced_t2w:
        try:
            # Create animated GIF showing slices through the T2w image
            gif2nif.write_gif_normal(t2_image)
        except Exception as e:
            print(f"Warning: Could not create GIF for {t2_image}: {e}")

    # Locate and move generated GIF files to BIDSonym directory
    if session is not None:
        # Look for GIFs in session-specific anatomical directory
        gif_search_path = opj(
            bids_dir, 
            f'sub-{subject_label}/ses-{session}/anat',
            f'sub-{subject_label}*.gif'
        )
    else:
        # Look for GIFs in subject's anatomical directory
        gif_search_path = opj(
            bids_dir, 
            f'sub-{subject_label}/anat',
            f'sub-{subject_label}*.gif'
        )
    
    list_gifs = glob(gif_search_path)

    # Ensure the bidsonym directory exists
    os.makedirs(bidsonym_path, exist_ok=True)

    # Move all generated GIF files to BIDSonym sourcedata directory
    for gif_file in list_gifs:
        try:
            # Move GIF from original location to organized sourcedata location
            move(gif_file, bidsonym_path)
            print(f"Moved GIF: {os.path.basename(gif_file)} to {bidsonym_path}")
        except Exception as e:
            print(f"Warning: Could not move GIF file {gif_file}: {e}")

    print(f"GIF generation completed for subject {subject_label}")
    if session:
        print(f"Session: {session}")


def create_graphics(bids_dir, subject_label, session=None, modalities=['T1w']):
    """
    Setup and run the graphics workflow which creates static plots and
    animated GIFs of defaced images for quality assessment.
    """

    # Validate modalities parameter
    supported_modalities = ['T1w', 'T2w', 'FLAIR']
    if not modalities or not isinstance(modalities, list):
        print("Warning: No valid modalities selected. Defaulting to ['T1w'].")
        modalities = ['T1w']
    
    # Filter to only supported modalities
    valid_modalities = [mod for mod in modalities if mod in supported_modalities]
    if not valid_modalities:
        print("Warning: No valid modalities found. Defaulting to ['T1w'].")
        valid_modalities = ['T1w']

    # Create Nipype workflow for graphics generation
    report_wf = pe.Workflow('report_wf')

    # Define input node with all required parameters
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bids_dir', 'subject_label', 'session', 'modalities']),
        name='inputnode'
    )
    
    # Create node to find original and defaced image pairs
    find_images = pe.Node(
        Function(
            input_names=['bids_dir', 'subject_label', 'session', 'modalities'],
            output_names=['original_images', 'defaced_images', 'output_paths'],
            function=find_image_pairs
        ),
        name='find_images'
    )
    
    # Create node for brain mask overlay plots
    plt_brainmask = pe.Node(
        Function(
            input_names=['bids_dir', 'subject_label', 'session', 't2w'],
            output_names=['t1w_files', 't2w_flag'],
            function=plot_brainmask_overlay
        ),
        name='plt_brainmask'
    )
    
    # Create node for before/after comparison plots
    plt_comparison = pe.MapNode(
        Function(
            input_names=['image', 'mask', 'outfile', 'bids_dir'],
            output_names=['out_file'],
            function=plot_defaced_comparison
        ),
        name='plt_comparison',
        iterfield=['image', 'mask', 'outfile']
    )
    
    # Create node for GIF generation
    gf_defaced = pe.Node(
        Function(
            input_names=['bids_dir', 'subject_label', 'session', 't2w'],
            function=gif_defaced
        ),
        name='gf_defaced'
    )

    # Connect the workflow nodes
    report_wf.connect([
        # Connect inputs to find_images node
        (inputnode, find_images, [
            ('bids_dir', 'bids_dir'),
            ('subject_label', 'subject_label'),
            ('modalities', 'modalities')
        ]),
        
        # Connect find_images output to comparison plots
        (find_images, plt_comparison, [
            ('original_images', 'image'),
            ('defaced_images', 'mask'),
            ('output_paths', 'outfile')
        ]),
        (inputnode, plt_comparison, [('bids_dir', 'bids_dir')]),
        
        # Connect inputs for brain mask overlay plots
        (inputnode, plt_brainmask, [
            ('bids_dir', 'bids_dir'),
            ('subject_label', 'subject_label')
        ]),
        
        # Connect inputs for GIF generation
        (inputnode, gf_defaced, [
            ('bids_dir', 'bids_dir'),
            ('subject_label', 'subject_label')
        ]),
    ])

    # Connect optional session input if provided
    if session:
        inputnode.inputs.session = session
        report_wf.connect([
            (inputnode, find_images, [('session', 'session')]),
            (inputnode, plt_brainmask, [('session', 'session')]),
            (inputnode, gf_defaced, [('session', 'session')]),
        ])

    # Set t2w parameter based on modalities
    t2w_requested = any(mod in ['T2w', 'FLAIR'] for mod in valid_modalities)
    plt_brainmask.inputs.t2w = t2w_requested if t2w_requested else None
    gf_defaced.inputs.t2w = t2w_requested if t2w_requested else None

    # Set all workflow inputs
    inputnode.inputs.bids_dir = bids_dir
    inputnode.inputs.subject_label = subject_label
    inputnode.inputs.modalities = valid_modalities
    
    # Display processing information
    print(f"Starting graphics workflow for subject {subject_label}")
    if session:
        print(f"Processing session: {session}")
    print(f"Processing modalities: {valid_modalities}")
    
    # Execute the complete workflow
    report_wf.run()
    print("Graphics workflow completed successfully")


def find_image_pairs(bids_dir, subject_label, session=None, modalities=['T1w']):
    """
    Find pairs of original and defaced images for comparison plotting.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS root directory.
    subject_label : str
        Label of subject (without 'sub-' prefix).
    session : str, optional
        Session label (without 'ses-' prefix).
    modalities : list
        List of modalities to process.
        
    Returns
    -------
    tuple
        (original_images, defaced_images, output_paths) - Lists of file paths
    """
    
    # Import required modules within function for Nipype compatibility
    import os
    from os.path import join as opj
    from bids import BIDSLayout
    
    # Initialize BIDS layout
    layout = BIDSLayout(bids_dir)
    
    # Define paths
    if session is not None:
        sourcedata_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}')
        output_base = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}')
    else:
        sourcedata_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}')
        output_base = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}')
    
    original_images = []
    defaced_images = []
    output_paths = []
    
    # Find image pairs for each modality
    for modality in modalities:
        # Find defaced images in main BIDS structure
        if session is not None:
            defaced_files = layout.get(
                subject=subject_label,
                session=session,
                suffix=modality,
                extension='nii.gz',
                return_type='filename'
            )
        else:
            defaced_files = layout.get(
                subject=subject_label,
                suffix=modality,
                extension='nii.gz',
                return_type='filename'
            )
        
        # Find corresponding original files in sourcedata
        for defaced_file in defaced_files:
            # Extract filename components
            basename = os.path.basename(defaced_file)
            original_filename = basename.replace('.nii.gz', '_desc-nondeid.nii.gz')
            
            # Look for original file
            original_file = opj(sourcedata_path, original_filename)
            
            if os.path.exists(original_file):
                original_images.append(original_file)
                defaced_images.append(defaced_file)
                
                # Create output path for comparison plot
                comparison_filename = basename.replace('.nii.gz', '_desc-comparison.png')
                output_path = opj(output_base, comparison_filename)
                output_paths.append(output_path)
    
    return original_images, defaced_images, output_paths


def plot_defaced_comparison(image, mask, outfile, bids_dir=None):
    """
    Plot defaced image with before/after comparison.

    Parameters
    ----------
    image : str
        Path to original image.
    mask : str
        Path to defaced/masked image.
    outfile : str
        Path for output plot.
    bids_dir : str, optional
        Path to BIDS root directory (for logging purposes).
    """
    
    # Import required modules within function for Nipype compatibility
    import matplotlib.pyplot as plt
    import numpy as np
    from nibabel import load
    
    # Load images
    orig_img = load(image)
    defaced_img = load(mask)
    
    # Get data arrays
    orig_data = orig_img.get_fdata()
    defaced_data = defaced_img.get_fdata()
    
    # Find middle slice in sagittal view (good for seeing face removal)
    mid_slice = orig_data.shape[0] // 2
    
    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(np.rot90(orig_data[mid_slice, :, :]), cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Plot defaced image
    ax2.imshow(np.rot90(defaced_data[mid_slice, :, :]), cmap='gray')
    ax2.set_title('Defaced')
    ax2.axis('off')
    
    # Add title
    fig.suptitle('Defacing Results Comparison')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    
    return outfile
