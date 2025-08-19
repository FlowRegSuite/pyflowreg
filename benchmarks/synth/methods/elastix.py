import numpy as np
import itk
import pathlib
import cv2
from scipy.ndimage import gaussian_filter


def normalize(img):
    """Normalize image to [0, 1] range matching reference."""
    img = img.astype(np.float32)
    img = img - np.amin(img)
    img = img / (np.amax(img) + 0.00001)
    return img


def imgaussfilt(img, sigma):
    """Apply Gaussian filter matching reference."""
    return gaussian_filter(img, sigma, mode='constant')


class DefaultProcessor:
    def __init__(self, sigma=None):
        self.sigma = sigma

    def process(self, frame):
        if self.sigma is None or self.sigma == 0:
            return normalize(frame.astype(np.float32))
        return normalize(imgaussfilt(frame, self.sigma).astype(np.float32))


class GradientProcessor:
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def process(self, frame):
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, 0)

        result = []
        for ch in frame:
            # central differences:
            f = normalize(imgaussfilt(ch, self.sigma))
            
            f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
            fx = f[1:-1, 2:] - f[1:-1, 0:-2]
            fx *= 0.5
            fy = f[2:, 1:-1] - f[0:-2, 1:-1]
            fy *= 0.5
            result.append(np.stack([fx, fy], 0))

        return np.concatenate(result, 0).astype(np.float32)


def estimate_flow(fixed, moving, params_path, preprocess=True, use_gradient=False):
    """
    Estimate optical flow using elastix, matching the reference implementation exactly.
    
    Args:
        fixed: Reference frame (H, W) or (H, W, C)
        moving: Moving frame (H, W) or (H, W, C)
        params_path: List of parameter file paths
        preprocess: If True, apply preprocessing
        use_gradient: If True, use gradient processor
    """
    # Handle multi-channel by taking channels separately
    if fixed.ndim == 3 and fixed.shape[-1] > 1:
        fixed_ch1 = fixed[..., 0]
        fixed_ch2 = fixed[..., 1]
        moving_ch1 = moving[..., 0]
        moving_ch2 = moving[..., 1]
        multi_channel = True
    else:
        if fixed.ndim == 3:
            fixed = fixed[..., 0]
        if moving.ndim == 3:
            moving = moving[..., 0]
        fixed_ch1 = fixed
        moving_ch1 = moving
        multi_channel = False
    
    # Apply preprocessing
    if preprocess:
        if use_gradient:
            # Use GradientProcessor for gradient configs
            processor = GradientProcessor(sigma=1.5)
            
            if multi_channel:
                # Process each channel to get gradients
                fixed_processed = processor.process(np.stack([fixed_ch1, fixed_ch2], 0))
                moving_processed = processor.process(np.stack([moving_ch1, moving_ch2], 0))
                
                # fixed_processed shape: (4, H, W) - [ch1_fx, ch1_fy, ch2_fx, ch2_fy]
                # Pass each gradient component as a separate scalar image
                f1 = itk.GetImageFromArray(fixed_processed[0].astype(np.float32))  # ch1_fx
                m1 = itk.GetImageFromArray(moving_processed[0].astype(np.float32))
                f2 = itk.GetImageFromArray(fixed_processed[1].astype(np.float32))  # ch1_fy
                m2 = itk.GetImageFromArray(moving_processed[1].astype(np.float32))
                f3 = itk.GetImageFromArray(fixed_processed[2].astype(np.float32))  # ch2_fx
                m3 = itk.GetImageFromArray(moving_processed[2].astype(np.float32))
                f4 = itk.GetImageFromArray(fixed_processed[3].astype(np.float32))  # ch2_fy
                m4 = itk.GetImageFromArray(moving_processed[3].astype(np.float32))
                
                # Multi-metric elastix registration
                p = itk.ParameterObject.New()
                for pfile in params_path:
                    p.AddParameterFile(str(pathlib.Path(pfile)))
                
                elastix_object = itk.ElastixRegistrationMethod.New(f1, m1,
                                                                  parameter_object=p,
                                                                  log_to_console=False,
                                                                  number_of_threads=12)
                elastix_object.AddFixedImage(f2)
                elastix_object.AddMovingImage(m2)
                elastix_object.AddFixedImage(f3)
                elastix_object.AddMovingImage(m3)
                elastix_object.AddFixedImage(f4)
                elastix_object.AddMovingImage(m4)
                elastix_object.UpdateLargestPossibleRegion()
                
                tx = elastix_object.GetTransformParameterObject()
            else:
                # Single channel gradient processing
                fixed_processed = processor.process(np.expand_dims(fixed_ch1, 0))
                moving_processed = processor.process(np.expand_dims(moving_ch1, 0))
                
                # fixed_processed shape: (2, H, W) - [fx, fy]
                # Pass each gradient component as a separate scalar image
                f1 = itk.GetImageFromArray(fixed_processed[0].astype(np.float32))  # fx
                m1 = itk.GetImageFromArray(moving_processed[0].astype(np.float32))
                f2 = itk.GetImageFromArray(fixed_processed[1].astype(np.float32))  # fy
                m2 = itk.GetImageFromArray(moving_processed[1].astype(np.float32))
                
                p = itk.ParameterObject.New()
                for pfile in params_path:
                    p.AddParameterFile(str(pathlib.Path(pfile)))
                
                elastix_object = itk.ElastixRegistrationMethod.New(f1, m1,
                                                                  parameter_object=p,
                                                                  log_to_console=False,
                                                                  number_of_threads=12)
                elastix_object.AddFixedImage(f2)
                elastix_object.AddMovingImage(m2)
                elastix_object.UpdateLargestPossibleRegion()
                
                tx = elastix_object.GetTransformParameterObject()
        else:
            # Use DefaultProcessor for non-gradient configs
            processor = DefaultProcessor(sigma=1.5)
            
            if multi_channel:
                # Process channels and use multi-metric registration
                fixed1_processed = processor.process(fixed_ch1)
                moving1_processed = processor.process(moving_ch1)
                fixed2_processed = processor.process(fixed_ch2)
                moving2_processed = processor.process(moving_ch2)
                
                f1 = itk.GetImageFromArray(fixed1_processed.astype(np.float32))
                m1 = itk.GetImageFromArray(moving1_processed.astype(np.float32))
                f2 = itk.GetImageFromArray(fixed2_processed.astype(np.float32))
                m2 = itk.GetImageFromArray(moving2_processed.astype(np.float32))
                
                p = itk.ParameterObject.New()
                for pfile in params_path:
                    p.AddParameterFile(str(pathlib.Path(pfile)))
                
                elastix_object = itk.ElastixRegistrationMethod.New(f1, m1,
                                                                  parameter_object=p,
                                                                  log_to_console=False,
                                                                  number_of_threads=12)
                elastix_object.AddFixedImage(f2)
                elastix_object.AddMovingImage(m2)
                elastix_object.UpdateLargestPossibleRegion()
                
                tx = elastix_object.GetTransformParameterObject()
            else:
                # Single channel intensity processing
                fixed_processed = processor.process(fixed_ch1)
                moving_processed = processor.process(moving_ch1)
                
                f = itk.GetImageFromArray(fixed_processed.astype(np.float32))
                m = itk.GetImageFromArray(moving_processed.astype(np.float32))
                
                p = itk.ParameterObject.New()
                for pfile in params_path:
                    p.AddParameterFile(str(pathlib.Path(pfile)))
                
                tx = itk.elastix_registration_method(f, m, parameter_object=p, log_to_console=False)
    else:
        # No preprocessing
        if fixed.ndim == 3 and fixed.shape[-1] > 1:
            fixed = fixed[..., 0]
        if moving.ndim == 3 and moving.shape[-1] > 1:
            moving = moving[..., 0]
            
        f = itk.GetImageFromArray(fixed.astype(np.float32))
        m = itk.GetImageFromArray(moving.astype(np.float32))
        
        p = itk.ParameterObject.New()
        for pfile in params_path:
            p.AddParameterFile(str(pathlib.Path(pfile)))
        
        tx = itk.elastix_registration_method(f, m, parameter_object=p, log_to_console=False)
    
    # Get displacement field using transformix
    # Need a reference image for the transformation
    ref_image = f if 'f' in locals() else (f1 if 'f1' in locals() else itk.GetImageFromArray(fixed.astype(np.float32)))
    d = itk.transformix_displacement_field(tx, ref_image)
    a = itk.GetArrayFromImage(d).astype(np.float32)
    
    # Return in (H, W, 2) format
    # ITK returns (y, x) so we swap to (x, y)
    v = np.stack([a[..., 1], a[..., 0]], axis=-1)
    
    return v