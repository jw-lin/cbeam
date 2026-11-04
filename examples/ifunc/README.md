# DM influence function for the batch examples

`ANDES_400pix_all_modes.fits` is the deformable-mirror modal basis the
`batch_pipeline` examples (`BatchedPiston.ipynb`, `python -m batch_pipeline.run`)
use to generate aberrated input pupils, included with its author's permission.

- 400 x 400 px pupil mask (an obstructed 6-petal pupil, 109,096 pupil pixels);
- 10 modes: 5 petal pistons (the sixth petal is the reference) and 5 smooth
  low-order modes.

The layout is the one `specula.data_objects.ifunc.IFunc.save` writes (HDU 1:
mode matrix stored as `(n_pixels, n_modes)`; HDU 2: mask), so it loads either
through specula or, without specula, through `batch_pipeline.SimpleIFunc`
(which needs astropy). `batch_pipeline.load_ifunc` picks whichever is
available; both give bit-identical results. Set `CBEAM_IFUNC_FILE` to use a
different file in the same format.
