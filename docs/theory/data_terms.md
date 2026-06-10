# Data Terms (Constancy Assumptions)

The variational solver estimates a displacement field $w = (u, v)$, where $u$
is the horizontal (x) and $v$ the vertical (y) component, by minimizing an
energy that combines a data term with a smoothness term
{cite}`brox2004high,flotho2022flow`. The data term encodes a *constancy
assumption*: a statement about which image property is preserved when a
structure moves between the reference and the current frame. PyFlowReg
implements three data terms, selected through the `constancy_assumption`
field of `OFOptions` (or the `const_assumption` parameter of
`pyflowreg.core.optical_flow.get_displacement`).

## Selecting a data term

| `OFOptions` value | Alias | Motion tensor function | Assumes constant |
|---|---|---|---|
| `"gc"` (default) | `"gradient"` | `get_motion_tensor_gc` | spatial intensity gradients |
| `"gray"` | `"brightness"` | `get_motion_tensor_gray` | pixel intensities |
| `"cs"` | `"census"` | `get_motion_tensor_cs` | local intensity ordering (census) |

```{literalinclude} ../snippets/theory/data_terms/select_data_term.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

The aliases `"gradient"`, `"brightness"`, and `"census"` are normalized to
the serialized values `"gc"`, `"gray"`, and `"cs"` when `OFOptions` validates
the field; `get_displacement` accepts both spellings directly.

Each data term is linearized into a per-pixel *motion tensor* with six
components $(J_{11}, J_{22}, J_{33}, J_{12}, J_{13}, J_{23})$, computed per
channel and combined with the channel `weight` in the solver. At every solver
update, the level solver evaluates the quadratic form
$s^2 = (du, dv, 1)\, J \,(du, dv, 1)^\top$ in the flow increments
$(du, dv)$ and penalizes it with the generalized Charbonnier function
$\rho_a(s^2) = (s^2 + \epsilon)^a$ {cite}`sun2010secrets`, using exponent
`a_data` for the data term and `a_smooth` for the smoothness term. See the
[Parameter Guide](parameters.md#diffusion-parameters) for those exponents and
the [core API reference](../api/core.md) for the motion tensor functions.

All three data terms are implemented by the native `flowreg` flow backend.
The `diso` backend supports only `"gc"` and raises a `ValueError` for
`"gray"`, `"cs"`, or any GNC setting; see
[Flow backends](../user_guide/backends.md).

## Brightness constancy (`"gray"`)

Brightness (gray-value) constancy is the classic optical flow assumption: the
grey value of a pixel does not change along its motion trajectory,
$I(x, y, t) = I(x + u, y + v, t + 1)$ {cite}`brox2004high`. The
implementation averages the spatial derivatives of the reference and the
(warped) moving frame and uses their difference as the temporal term, then
forms the motion tensor from these first-order derivatives.

This is the simplest of the three data terms, but it is susceptible to
brightness changes between frames {cite}`brox2004high`: any intensity change
that is not caused by motion is interpreted as motion. Choose `"gray"` when
intensity is stable between the frames and the reference.

## Gradient constancy (`"gc"`, default)

Gradient constancy assumes that the spatial intensity gradient $\nabla I$
remains constant under the displacement,
$\nabla I(x, y, t) = \nabla I(x + u, y + v, t + 1)$ {cite}`brox2004high`.
Because the constraint is built from image derivatives rather than raw
intensities, it is invariant to additive intensity offsets and therefore less
sensitive to brightness changes between frames than gray-value constancy
{cite}`brox2004high`, and it provides additional constraints in textured
regions.

The implementation forms the motion tensor from second-order spatial
derivatives and derivatives of the temporal difference, with per-pixel
normalization weights derived from the local gradient magnitudes. This is the
PyFlowReg default and the data term used by the MATLAB Flow-Registration
toolbox {cite}`flotho2022flow`; keep it unless you have a specific reason to
deviate.

## Census constancy (`"cs"`)

Census constancy assumes that the local *ordering* of intensities is
preserved: the census transform encodes each pixel by the signs of the
differences between its neighbors and itself, which makes the (hard)
transform invariant under monotonically increasing grey-value
transformations and the basis of an illumination-robust constancy assumption
{cite}`hafner2013census`.

The implementation uses a smoothed census transform over the eight-connected
3x3 neighborhood: for each neighbor offset, the directional difference
$r = (\text{neighbor} - \text{center}) / \text{dist}$ is mapped through a
smoothed Heaviside function
$s = \tfrac{1}{2}\left(1 + r / \sqrt{r^2 + \epsilon^2}\right)$, a
brightness-constancy-style tensor is built on the transformed signal, and
the tensors are averaged over the eight offsets. The smoothing width defaults
to $\epsilon = 0.1/255$, following the $\epsilon = 0.1$ convention of
{cite}`hafner2013census` for images scaled from $[0, 255]$ to approximately
$[0, 1]$; it is not exposed through `OFOptions` and can only be changed by
calling `get_motion_tensor_cs` directly.

Because of the smoothing and linearization, the invariance is approximate:
additive intensity offsets cancel exactly in the neighbor-center differences,
while multiplicative and other monotone intensity changes are handled only
approximately. Choose `"cs"` when intensity changes between frames violate
the brightness and gradient constancy assumptions; the trade-off is that
absolute intensity information is discarded in favor of local ordering within
the 3x3 neighborhood.

## Graduated non-convexity (GNC)

The generalized Charbonnier penalty $\rho_a(s^2) = (s^2 + \epsilon)^a$ is
convex in $s$ for $a \geq 0.5$ but not for smaller exponents, so with
sublinear exponents the energy can have multiple local minima. Graduated
non-convexity {cite}`blake1987visual` addresses this by first solving a
convex (quadratic) approximation of the problem and then progressively
deforming it toward the robust objective, warm-starting each stage from the
previous solution. Sun et al. apply this to optical flow by linearly
combining a quadratic with a robust objective, from fully quadratic to fully
robust {cite}`sun2010secrets`. In PyFlowReg, GNC is opt-in and off by
default.

### `gnc_schedule`

**Default value**: `None` (GNC off; the baseline single-pass solver is used)

A schedule is a 1D sequence of stage weights $\beta$. It is validated (both
by the `OFOptions` field validator and by
`pyflowreg.core.optical_flow.normalize_gnc_schedule`) to contain at least two
entries in $[0, 1]$ that are monotone nondecreasing, start at `0.0`, and end
at `1.0` — for example `(0.0, 0.5, 1.0)`. For each stage weight,
`get_displacement` runs a complete coarse-to-fine pyramid pass; each stage is
initialized with the displacement field of the previous stage (the first
stage with the user-supplied initialization, if any).

### Penalty blending

For a fixed stage weight $\beta$, the GNC level solver
(`compute_flow_gnc`) replaces the lagged data-term nonlinearity
$\psi = \rho_a'(s^2)$ of the baseline solver with

$$\psi_\beta = (1 - \beta) \cdot 1 + \beta \cdot \rho_a'(s^2),$$

the derivative of the blended penalty
$(1 - \beta)\, s^2 + \beta\, \rho_a(s^2)$ — a linear combination of the
quadratic and robust objectives as in {cite}`sun2010secrets`. At
$\beta = 0$ the data term is quadratic; at $\beta = 1$ it equals the robust
penalty of the baseline solver, so GNC changes the optimization path, not
the final model. The blend is applied only where the exponent lies in
$(0, 1)$: per channel for `a_data`, and for the smoothness term only when
`0 < a_smooth < 1`. With the default `a_smooth = 1.0` the smoothness term
remains linear diffusion and is unaffected by $\beta$.

### `warping_steps`

**Default value**: `None` (10 steps per level in GNC mode; ignored otherwise)

In GNC mode, every pyramid level performs `warping_steps` warp/relinearize
iterations: the moving image is warped with the current flow, the motion
tensors are rebuilt, the solver computes a flow increment, and a 5x5 median
filter is applied to the flow (on levels larger than 5 pixels in the smaller
dimension), following the intermediate median filtering and the 10 warping
steps per level of {cite}`sun2010secrets`. The value must be a positive
integer. Outside GNC mode the baseline path performs one linearization per
pyramid level and `warping_steps` is ignored.

### Practical guidance

- `gnc_schedule=None` (the default) preserves the established single-pass
  solver behavior; treat GNC as an opt-in refinement.
- Runtime grows accordingly: the number of level-solver calls increases from
  one per pyramid level to (number of stages) x `warping_steps` per level.
- `(0.0, 1.0)` is the minimal valid schedule; `(0.0, 0.5, 1.0)` adds an
  intermediate stage.
- GNC composes with all three data terms; the motion tensors are rebuilt with
  the selected data term at every warp step.
- The `diso` backend does not support GNC: resolving it through `OFOptions`
  with `gnc_schedule` or `warping_steps` set raises a `ValueError` (see
  [Flow backends](../user_guide/backends.md)).

## References

```{bibliography}
:filter: docname in docnames
```
