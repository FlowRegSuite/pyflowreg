import numpy as np
import itk
import pathlib
from scipy.ndimage import gaussian_filter


def preprocess_elastix(frame, apply_gaussian=True, sigma=1.0, compute_gradient=False):
    """
    Preprocess a frame for elastix matching the reference implementation.
    
    Args:
        frame: Input frame (H, W) or (H, W, C)
        apply_gaussian: If True, apply Gaussian filtering
        sigma: Gaussian filter sigma
        compute_gradient: If True, compute image gradients
    
    Returns:
        Preprocessed frame
    """
    # Handle multi-channel input
    if frame.ndim == 3 and frame.shape[-1] > 1:
        # Process first channel only (elastix typically uses single channel)
        frame = frame[..., 0]
    elif frame.ndim == 3:
        frame = frame[..., 0]
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    
    # Apply Gaussian filtering if requested
    if apply_gaussian:
        frame = gaussian_filter(frame, sigma, mode='constant')
    
    # Compute gradient if requested
    if compute_gradient:
        # Convert to ITK image for gradient computation
        itk_img = itk.GetImageFromArray(frame)
        gradient_filter = itk.GradientImageFilter.New(itk_img)
        gradient_filter.Update()
        gradient = itk.GetArrayFromImage(gradient_filter.GetOutput())
        return gradient
    
    return frame


def estimate_flow(fixed, moving, params_path, preprocess=True, use_gradient=False):
    """
    Estimate optical flow using elastix.
    
    Args:
        fixed: Reference frame (H, W) or (H, W, C)
        moving: Moving frame (H, W) or (H, W, C)
        params_path: List of parameter file paths
        preprocess: If True, apply preprocessing
        use_gradient: If True, register gradient images instead of intensities
    """
    # Apply preprocessing if requested
    if preprocess:
        # For gradient-based configs (matching reference elastix code)
        if use_gradient:
            # Apply Gaussian with sigma=1 then compute gradients
            fixed_processed = preprocess_elastix(fixed, apply_gaussian=True, sigma=1.0, compute_gradient=True)
            moving_processed = preprocess_elastix(moving, apply_gaussian=True, sigma=1.0, compute_gradient=True)
            
            # For multi-channel data, process each channel
            if fixed.ndim == 3 and fixed.shape[-1] > 1:
                fixed2 = preprocess_elastix(fixed[..., 1:2], apply_gaussian=True, sigma=1.0, compute_gradient=True)
                moving2 = preprocess_elastix(moving[..., 1:2], apply_gaussian=True, sigma=1.0, compute_gradient=True)
                
                # Use multi-metric registration with both gradient images
                f1 = itk.GetImageFromArray(fixed_processed.astype(np.float32))
                m1 = itk.GetImageFromArray(moving_processed.astype(np.float32))
                f2 = itk.GetImageFromArray(fixed2.astype(np.float32))
                m2 = itk.GetImageFromArray(moving2.astype(np.float32))
                
                # Setup multi-metric elastix
                p = itk.ParameterObject.New()
                for pfile in params_path:
                    p.AddParameterFile(str(pathlib.Path(pfile)))
                
                # Create elastix object with gradient images
                elastix_object = itk.ElastixRegistrationMethod.New(f1, m1,
                                                                  parameter_object=p,
                                                                  log_to_console=False,
                                                                  number_of_threads=12)
                elastix_object.AddFixedImage(f2)
                elastix_object.AddMovingImage(m2)
                elastix_object.UpdateLargestPossibleRegion()
                
                tx = elastix_object.GetTransformParameterObject()
            else:
                # Single channel gradient
                f = itk.GetImageFromArray(fixed_processed.astype(np.float32))
                m = itk.GetImageFromArray(moving_processed.astype(np.float32))
                
                p = itk.ParameterObject.New()
                for pfile in params_path:
                    p.AddParameterFile(str(pathlib.Path(pfile)))
                
                tx = itk.elastix_registration_method(f, m, parameter_object=p, log_to_console=False)
        else:
            # Standard intensity-based registration with Gaussian filtering
            fixed_processed = preprocess_elastix(fixed, apply_gaussian=True, sigma=1.5, compute_gradient=False)
            moving_processed = preprocess_elastix(moving, apply_gaussian=True, sigma=1.5, compute_gradient=False)
            
            f = itk.GetImageFromArray(fixed_processed.astype(np.float32))
            m = itk.GetImageFromArray(moving_processed.astype(np.float32))
            
            p = itk.ParameterObject.New()
            for pfile in params_path:
                p.AddParameterFile(str(pathlib.Path(pfile)))
            
            tx = itk.elastix_registration_method(f, m, parameter_object=p, log_to_console=False)
    else:
        # No preprocessing - use raw input
        if fixed.ndim == 3 and fixed.shape[-1] > 1:
            fixed = np.sqrt((fixed**2).sum(axis=-1))
        elif fixed.ndim == 3:
            fixed = fixed[..., 0]
            
        if moving.ndim == 3 and moving.shape[-1] > 1:
            moving = np.sqrt((moving**2).sum(axis=-1))
        elif moving.ndim == 3:
            moving = moving[..., 0]
        
        f = itk.GetImageFromArray(fixed.astype(np.float32))
        m = itk.GetImageFromArray(moving.astype(np.float32))
        
        p = itk.ParameterObject.New()
        for pfile in params_path:
            p.AddParameterFile(str(pathlib.Path(pfile)))
        
        tx = itk.elastix_registration_method(f, m, parameter_object=p, log_to_console=False)
    
    # Get displacement field
    d = itk.transformix_displacement_field(tx, f if 'f' in locals() else itk.GetImageFromArray(fixed.astype(np.float32)))
    a = itk.GetArrayFromImage(d).astype(np.float32)
    
    # Return in (H, W, 2) format
    # ITK returns (y, x) so we swap to (x, y)
    v = np.stack([a[..., 1], a[..., 0]], axis=-1)
    
    return v