Project name: pip2_segmentation

Author: Alessandro Ulivi (ale.ulivi@gmail.com)

Start date (yyyy/mm/dd): 2024/08/21

Status: ongoing (2025/01/14)

Description: the goal of the project is to implement the segmentation of pip2-enriched domains in the C. elegans early embryo. The project was started by Alessandro Ulivi at IGBMC (Illkirch-Graffenstaden, France). Pip2-enriched domains are fluorescently labelled structures. The labelling is obtained by the transgenic expression of a mCherry fluorescent tag (refer to the C. elegans transgenic line ACR074). Raw images have been acquired using a Nikon CSU-X1 spinning disk microscope, a 100x, 1.4 NA, oil immersion objective. The pixel size is 0.11x0.11 um. Raw images have been splat in train, validation and test dataset using the notebook organize_data_in_train_validation_test_folders.ipynb (for the moment, refer to its documentation).
The backbone of the project relies on the material from the course EMBO-DL4MIA (https://github.com/dl4mia)

The project is organized in the following packages and notebooks:
- checkpoints. Contains saved checkpoints for trained models. Checkpoints are saved whenever the metric of the validation data improved during model training (refer to run_training within train_model.py in modeltrain). Checkpoints can be re-loaded and used for test and predictions on new data. The name of the checkpoints is obtained using a timestamp (https://docs.python.org/3/library/datetime.html) at the beginning of the model's training, thus it indicates the exact day-time when the model started the training and it matches exactely the name of the data saved in "runs" for the model using Tensorboard Summary Writer.
- dataprep. Python package. Contains modules with the functions for data preparation (normalization, numpy-to-tensor transformation, data augmentation...)
- dependencies. Contins the .yml file which can be used to create the environment to run the project.
- license. Contains the file of the license regulating the use of the project.
- metrics. Python package. Contains modules with the functions for the models' loss functions and evaluation metrics of model's prediction.
- models. Python package. Contains modules with the models which can be trained and used for prediction (as of 2025/01/14 only a Unet CNN).
- modeltest. Python package. Contains modules with the function to test a model (trained or under training) on the validation and test data.
- modeltrain. Python package. Contains modules with the function to train a model using the train dataset (or the train and the validation data if the validation is done during training).
- runs. Contains the data saved using Tensorboard Summary Writer, which can be used to visualize the training process using chosen loss functions and metrics. These data also include images (both raw, target and prediction) of model trained or under training and the hyperparameters used for the model. The name of the data file is obtained using a timestamp (https://docs.python.org/3/library/datetime.html) at the beginning of the model's training, thus it indicates the exact day-time when the model (to which the data belong to) started training and it matches exactely the name of the model's checkpoint saved in "checkpoints".
- tests. Contains the data saved using Tensorboard Summary Writer, which can be used to visualize the test of the final trained model on the test dataset, using the the chosen loss function and metrics. These data also include images (both raw, target and prediction) of the tested model and the hyperparameters used for the model. The name of the data file is obtained using a timestamp (https://docs.python.org/3/library/datetime.html) at the beginning of the model's test, thus it indicates the exact day-time when the model (to which the data belong to) started the testing.
- utils. Python package. Contains modules with functions which serve multiple, miscellaneous functions, useful across multiple other packages and notebooks.
- organize_data_in_train_validation_test_folders.ipynb. The notebook used to split data into train, validation and test datasets. Refer to the internal documentation.
- plot_summary_graphs.ipynb. EARLY DEVELOPMENT. The notebook can be used to plot graph describing models metrics from the data saved using Tensorboard SummaryWriter and saved in "runs".
- run_test.ipynb. EARLY DEVELOPMENT. The notebook can be used to do test the final model on the test dataset.
- run_training_w_validation.ipynb. The notebook is used to train various models and tune their hyperparameters while validating the results on the validation dataset.
- segment_data.ipynb. EARLY DEVELOPMENT. The notebook can be used for prediction segmentation masks of new data after the training and test of the final model.
- test_data_preparation.ipynb. THIS NOTEBOOK WILL BE REMOVED IN THE FINAL VERSION OF PROJECT. The notebook is used to test the development of data preparation functions.
- test_training.ipynb. THIS NOTEBOOK WILL BE REMOVED IN THE FINAL VERSION OF PROJECT. The notebook is used for developing model training with validation.


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
- set device to "cuda" if gpu is available in run_training_w_validation.ipynb, in run_test.ipynb and in segment_data.ipynb by uncommenting the appropriate lines (and commenting out the line "device = torch.device("cpu")").

Roadmap:
2025/01/14. Need to:
1) complete the documentation of test_model within test_model.py (log_image_interv is not in the input variables, step is missing).
2) pass a proper step to test_model in run_test.ipynb.
3) change initial markdown of run_test.ipynb.
4) complete the documentation of DiceBCELoss and BCE_EdgeDiceLoss in metric.py (no description is present for the latter, no description of the weights is present for the first)
5) properly write the notebooks' markdowns.
6) document interpolate_curve in utils.py
7) check that the model works properly whit different depth, kernel size and padding (I can't remember if I did this already)
8) add bce_weight and and dice_weigth to dictionary saved by summary writer in run_training_w_validation notebook. 
9) don't save all the "empty" masks for the validation and test datasets, but only some of them
10) randomise the chunking of the images to avoid the bias of having pip2 enriched domains only at boarders
11) join train and validation datasets when generating them, avoiding masks with no labelled pixels, to form the dataset used for the final training

2025/01/15. Done:
8) add bce_weight and and dice_weigth to dictionary saved by summary writer in run_training_w_validation notebook.

2025/01/16. Need to:
12) check that loss function range implementation for DiceBCELoss and BCE_EdgeDiceLoss is working properly using the test_training and the run_training_w_validation notebooks.
13) implement a switch for whether or not weights and range should be used when training, as as they are at the moment, only DiceBCELoss and BCE_EdgeDiceLoss
loss functions can be used, which is rediculous. This includes introducing the switch in the train function within train_model.py. Passing this switch to
run_training and run_training_no_val functions also in train_model.py. Introduce the switch in test_training and run_training_w_validation notebooks, including
the saving of the parameter in the summary writer. To function properly the switch must allow working again with loss functions without weigths and ranges, as the DiceLoss and BCELoss.
14) Transfer the above changes (metric.py, train_model.py, run_training_w_validation) to Google Colab.
15) Test in Google Colab that the implementation of weights, range and switch works properly (when bce_weight=1, dice_weigth=1 and range_v=[0,2] I must obtain the results of 2024/11/12).
16) Implement the weights, loss function range and "weights switch" in test_model function within test_model.py and report it when it is used in run_training within train_model.py
17) Work on run_test notebook, to implement weights, loss_function_range and "weights switch", pass them properly to test_model and complete the proper encoding of the hypeparameters in the Summary Writer. 
18) Properly document crop_dimension and random_crop in utils_funct.py
19) transfer test_model.py changes (implementation of weights, loss function range and "weights switch") to Google Colab.

Done:
2) pass a proper step to test_model in run_test.ipynb. THIS WAS CORRECT, NO STEP SHOULD BE PASSED TO test_model IN run_test.ipynb.
12) check that loss function range implementation for DiceBCELoss and BCE_EdgeDiceLoss is working properly using the test_training and the run_training_w_validation notebooks. 
13) implement a switch for whether or not weights and range should be used when training, as as they are at the moment, only DiceBCELoss and BCE_EdgeDiceLoss
loss functions can be used, which is rediculous. This includes introducing the switch in the train function within train_model.py. Passing this switch to
run_training and run_training_no_val functions also in train_model.py. Introduce the switch in test_training and run_training_w_validation notebooks, including
the saving of the parameter in the summary writer. To function properly the switch must allow working again with loss functions without weigths and ranges, as the DiceLoss and BCELoss.
14) Transfer the above changes (metric.py, train_model.py, run_training_w_validation) to Google Colab.
16) Implement the weights, loss function range and "weights switch" in test_model function within test_model.py and report it when it is used in run_training within train_model.py
17) Work on run_test notebook, to implement weights, loss_function_range and "weights switch", pass them properly to test_model and complete the proper encoding of the hypeparameters in the Summary Writer. 
19) transfer test_model.py changes (implementation of weights, loss function range and "weights switch") to Google Colab.

