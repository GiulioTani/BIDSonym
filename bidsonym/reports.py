# Import all required modules at the top
import os
from datetime import datetime


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
    from glob import glob
    from os.path import join as opj
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from bids import BIDSLayout
    from nilearn.plotting import find_cut_slices, plot_stat_map

    print(f"DEBUG: plot_brainmask_overlay called for subject {subject_label}, session: {session}, t2w: {t2w}")

    # Initialize BIDS layout to query dataset structure
    layout = BIDSLayout(bids_dir)
    
    # Define paths - brain masks are in anat subdirectory
    if session is not None:
        anat_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}/anat')
        qc_output_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}/QC')
    else:
        anat_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/anat')
        qc_output_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/QC')

    print(f"DEBUG: Looking for brain masks in: {anat_path}")
    print(f"DEBUG: QC outputs will be saved to: {qc_output_path}")

    # Ensure output directories exist
    os.makedirs(qc_output_path, exist_ok=True)

    # List all files in anat directory to see what's available
    if os.path.exists(anat_path):
        all_files = os.listdir(anat_path)
        brain_mask_files = [f for f in all_files if 'brainmask' in f and '.nii.gz' in f]
        print(f"DEBUG: All files in anat directory: {all_files}")
        print(f"DEBUG: Brain mask files found in anat: {brain_mask_files}")
    else:
        print(f"DEBUG: Anat directory does not exist: {anat_path}")
        return (None, t2w)

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

    print(f"DEBUG: Found {len(defaced_t1w)} T1w images: {[os.path.basename(f) for f in defaced_t1w]}")

    # Process each T1w image found
    plots_created = 0
    for t1w in defaced_t1w:
        print(f"DEBUG: Processing T1w: {os.path.basename(t1w)}")
        
        # Updated brain mask filename construction to match the actual naming pattern
        t1w_basename = os.path.basename(t1w)
        brain_mask_filename = t1w_basename.replace('.nii.gz', '_brainmask_desc-nondeid.nii.gz')
        
        print(f"DEBUG: Looking for brain mask: {brain_mask_filename}")
        
        # Look for the brain mask in the anat directory
        brainmask_t1w = opj(anat_path, brain_mask_filename)
        
        if not os.path.exists(brainmask_t1w):
            print("DEBUG: Direct path not found, trying alternative patterns in anat directory...")
            
            # Try different naming patterns that might exist in anat directory
            alternative_patterns = [
                t1w_basename.replace('.nii.gz', '_brainmask.nii.gz'),
                t1w_basename.replace('.nii.gz', '_brain.nii.gz'),
                t1w_basename.replace('.nii.gz', '_mask.nii.gz'),
            ]
            
            found_mask = None
            for pattern in alternative_patterns:
                test_path = opj(anat_path, pattern)
                print(f"DEBUG: Testing pattern in anat: {pattern}")
                if os.path.exists(test_path):
                    found_mask = test_path
                    print(f"DEBUG: Found brain mask with alternative pattern: {pattern}")
                    break
            
            if found_mask:
                brainmask_t1w = found_mask
            else:
                # Try recursive search in anat directory with wildcards
                print("DEBUG: Trying recursive search for brain mask in anat directory...")
                recursive_patterns = [
                    f"*{t1w_basename.replace('.nii.gz', '')}*brainmask*.nii.gz",
                    f"*{t1w_basename.split('_')[0]}*brainmask*.nii.gz",
                ]
                
                found_recursive = False
                for pattern in recursive_patterns:
                    recursive_search = glob(opj(anat_path, pattern))
                    print(f"DEBUG: Recursive search in anat for '{pattern}': {recursive_search}")
                    if recursive_search:
                        brainmask_t1w = recursive_search[0]
                        print(f"DEBUG: Found brain mask via recursive search: {brainmask_t1w}")
                        found_recursive = True
                        break
                
                if not found_recursive:
                    print(f"DEBUG: No brain mask found for {t1w_basename}, skipping...")
                    continue
        else:
            print(f"DEBUG: Found brain mask: {brainmask_t1w}")
        
        try:
            # Create figure with subplots for three orthogonal views
            fig = figure(figsize=(15, 5))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.2, hspace=0)
            
            # Generate plots for each anatomical direction
            for i, direction in enumerate(['x', 'y', 'z']):
                ax = fig.add_subplot(1, 3, i + 1)
                
                # Find optimal slice positions for this direction
                cuts = find_cut_slices(t1w, direction=direction, n_cuts=12)
                
                # Plot brain mask overlaid on defaced T1w image
                plot_stat_map(
                    brainmask_t1w,
                    bg_img=t1w,
                    display_mode=direction,
                    cut_coords=cuts,
                    annotate=False,
                    dim=-1,
                    axes=ax,
                    colorbar=False
                )
            
            # Save the plot to QC directory
            output_filename = t1w_basename.replace('.nii.gz', '_desc-brainmaskdeid.png')
            output_path = opj(qc_output_path, output_filename)
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plots_created += 1
            print(f"DEBUG: Saved brain mask overlay plot: {output_path}")
            
        except Exception as e:
            print(f"ERROR: Failed to create brain mask overlay for {t1w}: {e}")
            import traceback
            traceback.print_exc()
            plt.close()

    # Process T2w/FLAIR images if requested
    if t2w is not None:
        print(f"DEBUG: Processing FLAIR images (t2w={t2w})")
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

        print(f"DEBUG: Found {len(defaced_flair)} FLAIR images: {[os.path.basename(f) for f in defaced_flair]}")

        # Process each FLAIR image found
        for flair in defaced_flair:
            print(f"DEBUG: Processing FLAIR: {os.path.basename(flair)}")
            
            # Similar brain mask search logic for FLAIR in anat directory
            flair_basename = os.path.basename(flair)
            brain_mask_filename = flair_basename.replace('.nii.gz', '_brainmask_desc-nondeid.nii.gz')
            
            brainmask_flair = opj(anat_path, brain_mask_filename)
            
            if not os.path.exists(brainmask_flair):
                # Try alternative patterns for FLAIR in anat directory
                alternative_patterns = [
                    flair_basename.replace('.nii.gz', '_brainmask.nii.gz'),
                    flair_basename.replace('.nii.gz', '_brain.nii.gz'),
                    flair_basename.replace('.nii.gz', '_mask.nii.gz'),
                ]
                
                found_mask = None
                for pattern in alternative_patterns:
                    test_path = opj(anat_path, pattern)
                    if os.path.exists(test_path):
                        found_mask = test_path
                        break
                
                if found_mask:
                    brainmask_flair = found_mask
                else:
                    recursive_search = glob(opj(anat_path, f"*{flair_basename.replace('.nii.gz', '')}*brainmask*.nii.gz"))
                    if recursive_search:
                        brainmask_flair = recursive_search[0]
                    else:
                        print(f"DEBUG: No brain mask found for FLAIR {flair_basename}, skipping...")
                        continue
            
            try:
                # Same plotting logic as T1w
                fig = figure(figsize=(15, 5))
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.2, hspace=0)
                
                for i, direction in enumerate(['x', 'y', 'z']):
                    ax = fig.add_subplot(1, 3, i + 1)
                    cuts = find_cut_slices(flair, direction=direction, n_cuts=12)
                    
                    plot_stat_map(
                        brainmask_flair,
                        bg_img=flair,
                        display_mode=direction,
                        cut_coords=cuts,
                        annotate=False,
                        dim=-1,
                        axes=ax,
                        colorbar=False
                    )
                
                # Save FLAIR plot to QC directory
                output_filename = flair_basename.replace('.nii.gz', '_desc-brainmaskdeid.png')
                output_path = opj(qc_output_path, output_filename)
                
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                plots_created += 1
                print(f"DEBUG: Saved FLAIR brain mask overlay plot: {output_path}")
                
            except Exception as e:
                print(f"ERROR: Failed to create FLAIR brain mask overlay for {flair}: {e}")
                import traceback
                traceback.print_exc()
                plt.close()

    print(f"DEBUG: Created {plots_created} brain mask overlay plots total")
    
    # Return processed files
    return (defaced_t1w[0] if defaced_t1w else None, t2w)


def gif_defaced(bids_dir, subject_label, session=None, t2w=None):
    """
    Create animated GIFs that loop through slices of defaced images in
    orthogonal directions (x, y, z).
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
    
    # Define paths - GIFs should go to session-aware QC directory
    if session is not None:
        qc_output_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}/QC')
    else:
        qc_output_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/QC')

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

    # Locate generated GIF files in the anat directory where they're created
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

    # Ensure the session-aware QC directory exists
    os.makedirs(qc_output_path, exist_ok=True)

    # Move all generated GIF files to session-aware QC directory
    for gif_file in list_gifs:
        try:
            # Move GIF to session-aware QC directory
            move(gif_file, qc_output_path)
            print(f"Moved GIF: {os.path.basename(gif_file)} to {qc_output_path}")
        except Exception as e:
            print(f"Warning: Could not move GIF file {gif_file}: {e}")

    print(f"GIF generation completed for subject {subject_label}")
    if session:
        print(f"Session: {session}")


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
    
    # Define paths - original images are in anat subdirectory, outputs go to QC
    if session is not None:
        sourcedata_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}/anat')
        output_base = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/ses-{session}/QC')
    else:
        sourcedata_path = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/anat')
        output_base = opj(bids_dir, f'sourcedata/bidsonym/sub-{subject_label}/QC')
    
    # Ensure QC directory exists
    os.makedirs(output_base, exist_ok=True)
    
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
        
        print(f"DEBUG: Found {len(defaced_files)} defaced {modality} files")
        
        # Find corresponding original files in sourcedata anat directory
        for defaced_file in defaced_files:
            # Extract filename components
            basename = os.path.basename(defaced_file)
            original_filename = basename.replace('.nii.gz', '_desc-nondeid.nii.gz')
            
            # Look for original file in anat directory
            original_file = opj(sourcedata_path, original_filename)
            
            print(f"DEBUG: Looking for original file: {original_file}")
            
            if os.path.exists(original_file):
                original_images.append(original_file)
                defaced_images.append(defaced_file)
                
                # Create output path for comparison plot in QC directory
                comparison_filename = basename.replace('.nii.gz', '_desc-comparison.png')
                output_path = opj(output_base, comparison_filename)
                output_paths.append(output_path)
                
                print(f"DEBUG: Added image pair: {os.path.basename(original_file)} -> {os.path.basename(defaced_file)}")
            else:
                print(f"DEBUG: Original file not found: {original_filename}")
    
    print(f"DEBUG: Found {len(original_images)} image pairs total")
    return original_images, defaced_images, output_paths


def plot_defaced_comparison(image, mask, outfile, bids_dir=None):
    """
    Plot before/after comparison of defacing results.
    Shows original image on left and defaced image on right with single sagittal cut.
    
    Parameters
    ----------
    image : str
        Path to original (non-defaced) image.
    mask : str
        Path to defaced image.
    outfile : str
        Path for output comparison plot.
    bids_dir : str, optional
        BIDS directory path (for compatibility).
        
    Returns
    -------
    str
        Path to the created output file.
    """
    
    # Import all required modules within the function for Nipype compatibility
    import os
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from nilearn.plotting import plot_anat
    import nibabel as nib
    
    print("DEBUG: Creating comparison plot for:")
    print(f"  Original: {os.path.basename(image) if image else 'None'}")
    print(f"  Defaced: {os.path.basename(mask) if mask else 'None'}")
    print(f"  Output: {os.path.basename(outfile) if outfile else 'None'}")
    
    # Validate inputs
    if not image or not os.path.exists(image):
        print(f"ERROR: Original image not found or not provided: {image}")
        return outfile
    
    if not mask or not os.path.exists(mask):
        print(f"ERROR: Defaced image not found or not provided: {mask}")
        return outfile
    
    if not outfile:
        print("ERROR: Output file path not provided")
        return outfile
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    try:
        # Load the original image to find the middle sagittal slice
        img_nib = nib.load(image)
        img_shape = img_nib.shape
        
        # Calculate middle sagittal slice (x-coordinate)
        middle_x = img_shape[0] // 2
        
        # Create figure with two subplots side by side
        fig = figure(figsize=(12, 6))
        
        # Plot original image on the left (sagittal view at middle slice)
        ax1 = fig.add_subplot(1, 2, 1)
        plot_anat(
            image, 
            display_mode='x', 
            cut_coords=[middle_x],
            axes=ax1, 
            title='Original', 
            annotate=False,
            draw_cross=False
        )
        
        # Plot defaced image on the right (sagittal view at same middle slice)
        ax2 = fig.add_subplot(1, 2, 2)
        plot_anat(
            mask, 
            display_mode='x', 
            cut_coords=[middle_x],
            axes=ax2, 
            title='Defaced', 
            annotate=False,
            draw_cross=False
        )
        
        # Add overall title with subject information
        subject_info = os.path.basename(image).split('_')[0]  # Extract subject info
        fig.suptitle(f'Defacing Comparison - {subject_info}', fontsize=14)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"DEBUG: Saved comparison plot: {outfile}")
        return outfile
        
    except Exception as e:
        print(f"ERROR: Failed to create comparison plot: {e}")
        import traceback
        traceback.print_exc()
        plt.close()
        return outfile


def create_graphics(bids_dir, subject_label, session=None, modalities=['T1w']):
    """
    Setup and run the graphics workflow which creates static plots and
    animated GIFs of defaced images for quality assessment.
    """

    # Import required modules within function for Nipype compatibility
    from nipype.interfaces import utility as niu
    from nipype.interfaces.utility import Function
    import nipype.pipeline.engine as pe

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

    print(f"Starting graphics workflow for subject {subject_label}")
    if session:
        print(f"Processing session: {session}")
    print(f"Processing modalities: {valid_modalities}")

    # Find image pairs for comparison
    original_images, defaced_images, output_paths = find_image_pairs(
        bids_dir, subject_label, session, valid_modalities
    )
    
    if not original_images:
        print(f"No image pairs found for comparison for subject {subject_label}")
        if session:
            print(f"Session: {session}")
    else:
        print(f"Creating {len(original_images)} comparison plots...")
        
        # Create comparison plots using Nipype workflow
        comparison_wf = pe.Workflow('comparison_plots')
        
        # Create iterables for processing multiple image pairs
        inputnode = pe.Node(niu.IdentityInterface([
            'original_images', 
            'defaced_images', 
            'output_paths'
        ]), name='inputnode')
        
        # Set the input lists
        inputnode.inputs.original_images = original_images
        inputnode.inputs.defaced_images = defaced_images
        inputnode.inputs.output_paths = output_paths
        
        # Create plot comparison node with proper iterfields
        plot_node = pe.MapNode(
            Function(
                input_names=['image', 'mask', 'outfile', 'bids_dir'],
                output_names=['out_file'],
                function=plot_defaced_comparison
            ),
            name='plt_comparison',
            iterfield=['image', 'mask', 'outfile']
        )
        
        # Set the bids_dir input
        plot_node.inputs.bids_dir = bids_dir
        
        # Connect the workflow
        comparison_wf.connect([
            (inputnode, plot_node, [
                ('original_images', 'image'),
                ('defaced_images', 'mask'),
                ('output_paths', 'outfile')
            ])
        ])
        
        # Run the workflow
        try:
            comparison_wf.run()
            print(f"Successfully created {len(original_images)} comparison plots")
        except Exception as e:
            print(f"ERROR: Failed to create comparison plots: {e}")
            import traceback
            traceback.print_exc()

    # Create brain mask overlay plots
    print("Creating brain mask overlay plots...")
    try:
        t2w_requested = any(mod in ['T2w', 'FLAIR'] for mod in valid_modalities)
        plot_brainmask_overlay(bids_dir, subject_label, session, t2w_requested if t2w_requested else None)
    except Exception as e:
        print(f"ERROR: Failed to create brain mask overlays: {e}")
        import traceback
        traceback.print_exc()

    # Create animated GIFs
    print("Creating animated GIFs...")
    try:
        t2w_requested = any(mod in ['T2w', 'FLAIR'] for mod in valid_modalities)
        gif_defaced(bids_dir, subject_label, session, t2w_requested if t2w_requested else None)
    except Exception as e:
        print(f"ERROR: Failed to create GIFs: {e}")
        import traceback
        traceback.print_exc()

    print("Graphics workflow completed successfully")
