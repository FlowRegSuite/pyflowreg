"""
Sequential executor - processes frames one by one without parallelization.
"""

from typing import Callable, Tuple
import numpy as np
from .base import BaseExecutor


class SequentialExecutor(BaseExecutor):
    """
    Sequential executor that processes frames one at a time.
    
    This is the simplest executor and serves as a reference implementation.
    It's also the most memory-efficient as it only processes one frame at a time.
    """
    
    def __init__(self, n_workers: int = 1):
        """
        Initialize sequential executor.
        
        Args:
            n_workers: Ignored for sequential executor, always uses 1.
        """
        super().__init__(n_workers=1)
    
    def process_batch(
        self,
        batch: np.ndarray,
        batch_proc: np.ndarray,
        reference_raw: np.ndarray,
        reference_proc: np.ndarray,
        w_init: np.ndarray,
        get_displacement_func: Callable,
        imregister_func: Callable,
        interpolation_method: str = 'cubic',
        progress_callback: Callable = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process frames sequentially.
        
        Args:
            batch: Raw frames to register, shape (T, H, W, C)
            batch_proc: Preprocessed frames for flow computation, shape (T, H, W, C)
            reference_raw: Raw reference frame, shape (H, W, C)
            reference_proc: Preprocessed reference frame, shape (H, W, C)
            w_init: Initial flow field, shape (H, W, 2)
            get_displacement_func: Function to compute optical flow
            imregister_func: Function to apply flow field for registration
            interpolation_method: Interpolation method for registration
            **kwargs: Additional parameters including 'flow_params' dict
            
        Returns:
            Tuple of (registered_frames, flow_fields)
        """
        T, H, W, C = batch.shape
        
        # Get flow parameters from kwargs
        flow_params = kwargs.get('flow_params', {})
        
        # Initialize output arrays (use empty instead of zeros for performance)
        registered = np.empty_like(batch)
        flow_fields = np.empty((T, H, W, 2), dtype=np.float32)
        
        # Process each frame sequentially
        for t in range(T):
            # Compute optical flow for this frame with all parameters
            flow = get_displacement_func(
                reference_proc, 
                batch_proc[t], 
                uv=w_init.copy(),
                **flow_params
            )
            
            # Apply flow field to register the frame
            reg_frame = imregister_func(
                batch[t],
                flow[..., 0],
                flow[..., 1],
                reference_raw,
                interpolation_method=interpolation_method
            )
            
            # Store results
            flow_fields[t] = flow.astype(np.float32, copy=False)
            
            # Handle case where registered frame might have fewer channels
            if reg_frame.ndim < registered.ndim - 1:
                registered[t, ..., 0] = reg_frame
            else:
                registered[t] = reg_frame
            
            # Call progress callback for this frame
            if progress_callback is not None:
                progress_callback(1)
        
        return registered, flow_fields
    
    def get_info(self) -> dict:
        """Get information about this executor."""
        info = super().get_info()
        info.update({
            'parallel': False,
            'description': 'Sequential frame-by-frame processing'
        })
        return info


# Register this executor with RuntimeContext on import
SequentialExecutor.register()