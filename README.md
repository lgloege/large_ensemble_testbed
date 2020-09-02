<h1 align="center"> Large Ensemble Testbed </h1>

<p align="center">
   <img height="400" src="https://github.com/lgloege/large_ensemble_testbed/blob/master/figures/figure1.png">
</p>


This repo contains code used in the manuscript ["Quantifying errors in observationally-based estimates of ocean carbon sink variability"](https://www.essoar.org/doi/10.1002/essoar.10502036.2). The Large Ensemble Testbed (LET) is available [here](https://figshare.com/collections/Large_ensemble_pCO2_testbed/4568555)

## Abstract
Reducing uncertainty in the global carbon budget requires better quantification of ocean CO2 uptake and its temporal variability. Several methodologies for reconstructing air-sea CO2 exchange from sparse pCO2 observations indicate larger decadal variability than estimated using ocean models. We develop a new application of multiple Large Ensemble Earth system models to assess these reconstructions’ ability to estimate spatiotemporal variability. With our Large Ensemble Testbed, pCO2 fields from 25 ensemble members each of four independent Earth system models are subsampled as the observations and the reconstruction is performed as it would be with real- world observations. The power of a testbed is that the perfect reconstruction is known for each of the 100 original model fields; thus, reconstruction skill can be comprehensively assessed. We find that a commonly used neural-network approach can skillfully reconstruct air-sea CO2 fluxes when and where it is trained with sufficient data. Flux bias is low for the global mean and Northern Hemisphere, but can be regionally high in the Southern Hemisphere. The phase and amplitude of the seasonal cycle are accurately reconstructed outside of the tropics, but longer-term variations are reconstructed with only moderate skill. For Southern Ocean decadal variability, insufficient sampling leads to a 39% [15%:58%, interquartile range] overestimation of amplitude, and phasing is only moderately correlated with known truth (r=0.54 [0.46:0.63]). Globally, the amplitude of decadal variability is overestimated by 21% [3%:34%]. Machine learning, when supplied with sufficient data, can skillfully reconstruct ocean properties. However, data sparsity remains a fundamental limitation to quantification of decadal variability in the ocean carbon sink.


##  notebooks
This contains jupyter notebooks used to generate figures.
Figures were finalized in illustrator

## scripts
This contains auxillary scripts for this project

