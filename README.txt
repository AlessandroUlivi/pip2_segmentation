Project name: pip2_segmentation

Author: Alessandro Ulivi (ale.ulivi@gmail.com)

Start date (yyyy/mm/dd): 2024/08/21

Status: ongoing (2024/10/24)

Description: the goal of the project is to implement the segmentation of pip2-enriched domains in the C. elegans early embryo. The project was started by Alessandro Ulivi at IGBMC (Illkirch-Graffenstaden, France). Pip2-enriched domains are fluorescently labelled structures. The labelling is obtained by the transgenic expression of a mCherry fluorescent tag (refer to the C. elegans transgenic line ACR074). Images have been acquired using a Nikon CSU-X1 spinning disk microscope, a 100x, 1.4 NA, oil immersion objective. The pixel size is 0.11x0.11 um.
The backbone of the project relies on the material from the course EMBO-DL4MIA (https://github.com/dl4mia)

Dependencies: The pip2_segmentation.yml allows to create an environment with all the dependencies to run the scripts. The present package versions were used to establish the pipeline:
- python 3.12.5
- jupyterlab 4.2.5
- pip 24.2
- numpy 1.26.4 (required for compatibility with tensorboard)
- matplotlib 3.9.2
- pandas 2.2.2
- scipy 1.14.1
- scikit-image 0.24.0
- tifffile 2024.08.30
- scikit-learn 1.5.1
- seaborn 0.13.2
- cpuonly 2.0
- pytorch 2.4.0
- torchvision 0.19.0
- torchaudio 2.4.0
- tensorboard 2.17.0
- roifile 2024.5.24
- kornia 0.7.4

NOTE: the environment (package cpuonly - pytorch Build py3.12_cpu_0) and the notebooks are set up to run on cpu.
In order to run the scripts with gpu support:
- remove the cpuonly package from the environment.
- set device to "cuda" if gpu is available in run_training_w_validation.ipynb and in run_test.ipynb by uncommenting the appropriate lines (and commenting out "device = torch.device("cpu")").

Roadmap:
2025/01/14. Need to:
1) complete the documentation of test_model within test_model.py (log_image_interv is not in the input variables, step is missing).
2) pass a proper step to test_model in run_test.ipynb.
3) change initial markdown of run_test.ipynb.
4) complete the documentation of DiceBCELoss and BCE_EdgeDiceLoss in metric.py (no description is present for the latter, no description of the weights is present for the first)
5) properly write the notebooks' markdowns.
