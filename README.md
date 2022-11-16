# Master of Science Project 2022/2023

URL to assignment: https://www.overleaf.com/read/tzwckfrpfbvw

Current best performing reconstruction module with score 0.0016208146698772907:
![](../TEMP/ACP/7_combined_pred.jpg)
../TEMP/ACP/7_combined_pred.jpg

---
# Blog:
### Tuesday 25.10.2022
I have implemented the multi-output architecture of the model. I have started to test out the multi output prediction using the acoustic impedance values from TNW. Currently, my laptop is not nearly powerful enough to run these models. Even in 1D. I have been talking with IT, both at NGI and NTNU, and tomorrow I'm getting access to a powerful desktop computer which I will remote access using my computer. This way, It will be possible to run computations at any time. This will be good because it will let optuna get the time to zero in on good model parameters. The architecture is essentially finished, only lacking minor tweaks for dimension tolerance and input feature amount tolerance. Some of the minor other funcitons for visualizations and statistics require some update for the new multi-output model, but this should not take too much effort.

### Wednesday 19.10.2022
I realize its been a while sist I gave an update last. I have implemented a script that lets me permute different model configurations. This works with a python generator that I use to give the models systematic names. I also implemented using optuna, which automatically adjusts the hyperparameters. This also works with the systematic naming. At present, I have a model which takes in the seismic image and outputs a similar looking image reliably with a loss of approx 1.4. Soon I'm going to start to predict on acoustic impedance, since that dataset is exhaustive. After that I'm going to experiment on using the feature recognition of these models on predicting CPT parameters. Academics wise, I'm going to start by justifying the prediction of acoustic impedance geologically. The leap to prediction of CPT parameters heavily relies on the ability to infer information about CPT values both from recognizing geological features (texturally), and correlating with the amplitude of the seismic signal.

### Monday 03.10.2022
I have ran a couple of models now, and have been somewhat able to overfit to a training image. I'm planning to run optuna this week to see how well I can have the TCN perform. After trying to upload the ML-weights to the repository I encountered some issues with pushing it. I think the file size is too big for the models, and that they will need to be downloaded seperately. If you are reading this it means that I sorted out the issue, and was able to push to the repo. 

### Monday 19.09.2022
The plan this week is to start running models and debugging the tcn code. Although these preliminary models will probably not give any good results, I will create them in order to get some hands-on experience with using and parameterizing the tcn network. I worked on loading the segy data into python, and writing code that segments the data into different images of the same size. All images in the training data must be of the same shape, so I need to find out a good depth cut-off for the data. I will start off with using one single inline seismic set of traces (TNW_B02_5110), training on images from the left half and testing on images of the same size on the right.

### Thursday 08.09.2022
This week I have started implementing the 2D and 1D TCN architectures. They are not finished, but I'm getting there. I have experimented with linux and running shell scripts on Odin. Ideally I write all instructions on my laptop, and then run a script which makes sure to dump all conda changes to the yaml and push my python code to the repo. After that it will have odin execute a script that pulls the changes, updates the conda environment and creates a new git branch. In the new branch Odin will start running all the models. After the models are ran, Odin will push to the new branch and send out an Email with some short summary information. I will then manually review and merge the branch into main. For performance review, I will use the reconstruction module to see where the model finds it hardest to reconstruct the seismic image.

### Wednesday 31.08.2022
I have been reading the Integrated ground model report from the TNWWFZ project, and gotten more familliar. Meanwhile I have been preparing for the first meeting of the semester.

### Thursday 25.08.2022
This past week I have started to set up the necessary infrastructure that should be in place to work on the MSc assignment. Among these are the github repository, python miniconda environment (for which a yml file will be added to the repo soon), Overleaf document using NTNU's recommended typesetting and structure. I started writing the log script, which will allow me to store the ML weights from any given training situation so that they may be loaded at any time I should want to access them. Each training instance should create a subfolder within the Models folder, and include some descriptive text document.
