import numpy as np
import itk
import pathlib


def estimate_flow(fixed, moving, params_path):
    # For multichannel/gradient inputs, use gradient magnitude
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
    
    tx = itk.elastix_registration_method(f, m, parameter_object=p)
    d = itk.transformix_displacement_field(tx, f)
    a = itk.GetArrayFromImage(d).astype(np.float32)
    
    # Return in (H, W, 2) format
    # ITK returns (y, x) so we swap to (x, y)
    v = np.stack([a[..., 1], a[..., 0]], axis=-1)
    
    return v