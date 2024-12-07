import numpy as np
from my_package import get_displacement  # Assuming get_displacement is implemented as discussed

def main():
    # Create or load two frames as numpy arrays
    # For this example, let's just create synthetic frames.
    # Suppose m=64, n=64, single channel
    m, n = 64, 64
    fixed = np.zeros((m, n), dtype=np.float64)
    moving = np.zeros((m, n), dtype=np.float64)

    # Introduce a simple synthetic shift in 'moving' frame
    # e.g., shift one pixel to the right
    moving[:, 1:] = 1.0

    # Compute displacement field
    du, dv = get_displacement(
        fixed,
        moving,
        alpha=(2,2),
        iterations=20,
        update_lag=10,
        a_data=0.45,
        a_smooth=0.5,
        hx=1.0,
        hy=1.0
    )

    # Print or analyze the results
    # du, dv are displacement fields in the vertical (u) and horizontal (v) directions respectively
    magnitude = np.sqrt(du**2 + dv**2)
    print("Displacement magnitude:", magnitude.mean())
    print("U-field mean:", du.mean())
    print("V-field mean:", dv.mean())

if __name__ == "__main__":
    main()