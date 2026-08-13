This repository provides the reconstruction code for our snapshot hyperspectral imaging system based on spectral–spatial compressed sensing. A single frame captured by the compressive-readout CMOS image sensor (CS-CIS) carries nine spectrally filtered sub-images, further compressed along the row direction during readout. From this doubly compressed measurement, the code recovers a hyperspectral datacube of 230 bands over 505–734 nm, corresponding to a total compression rate of 0.98%.

The pipeline consists of three steps, to be run in order from the repository root:

  1. Code/Spatial_Recovery.py — reconstructs the full sensor-plane image from the row-compressed frame via plug-and-play ADMM with the pretrained DRUNet denoiser.
  2. Code/Align.m — detects the nine sub-images by normalized cross-correlation and stacks them in the calibrated filter order.
  3. Code/Spectral_Recovery.m — decodes each pixel into the full spectrum through non-negative ℓ1-regularized least squares on a Gaussian sparsifying dictionary.
  Sample measurements are included in Data/. The pretrained DRUNet weights (drunet_gray.pth) should be placed in model_zoo/.
  
The l1_ls solver was developed by Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd (https://web.stanford.edu/~boyd/l1_ls/), and the DRUNet architecture and weights are from Kai Zhang's KAIR (https://github.com/cszn/KAIR). We thank the original authors for making their work openly available.
