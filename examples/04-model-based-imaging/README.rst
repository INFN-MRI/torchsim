Model-based imaging
-------------------

Reconstructing parameter maps from k-space without forming the contrast images
in between.

A quantitative scan is usually reconstructed twice: once to make one image per
contrast, and again -- voxel by voxel -- to turn those images into maps. The
first step has no idea what the second one is for. Writing the signal model
into the forward operator removes the intermediate step, and the echoes then
constrain one another instead of being recovered separately.

There are two ways to do it. A **linear subspace** writes the signal in the
low-rank basis the train spans and reconstructs the coefficients, which stays
linear and so has no local minima and no starting guess. **Nonlinear inversion**
keeps the model itself inside the operator and solves for the maps directly,
which costs more and is what a signal with no small basis needs.
