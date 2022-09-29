# Master of Science Project 2022/2023

URL to assignment: https://www.overleaf.com/read/tzwckfrpfbvw

Current best performing reconstruction module:
![](/TEMP/pred.jpg)

*/TEMP/seismic_images/pred.jpg*

# Blog:
### Monday 19.09.2022
The plan this week is to start running models and debugging the tcn code. Although these preliminary models will probably not give any good results, I will create them in order to get some hands-on experience with using and parameterizing the tcn network. I worked on loading the segy data into python, and writing code that segments the data into different images of the same size. All images in the training data must be of the same shape, so I need to find out a good depth cut-off for the data. I will start off with using one single inline seismic set of traces (TNW_B02_5110), training on images from the left half and testing on images of the same size on the right.

### Thursday 08.09.2022
This week I have started implementing the 2D and 1D TCN architectures. They are not finished, but I'm getting there. I have experimented with linux and running shell scripts on Odin. Ideally I write all instructions on my laptop, and then run a script which makes sure to dump all conda changes to the yaml and push my python code to the repo. After that it will have odin execute a script that pulls the changes, updates the conda environment and creates a new git branch. In the new branch Odin will start running all the models. After the models are ran, Odin will push to the new branch and send out an Email with some short summary information. I will then manually review and merge the branch into main. For performance review, I will use the reconstruction module to see where the model finds it hardest to reconstruct the seismic image.

### Wednesday 31.08.2022
I have been reading the Integrated ground model report from the TNWWFZ project, and gotten more familliar. Meanwhile I have been preparing for the first meeting of the semester.

### Thursday 25.08.2022
This past week I have started to set up the necessary infrastructure that should be in place to work on the MSc assignment. Among these are the github repository, python miniconda environment (for which a yml file will be added to the repo soon), Overleaf document using NTNU's recommended typesetting and structure. I started writing the log script, which will allow me to store the ML weights from any given training situation so that they may be loaded at any time I should want to access them. Each training instance should create a subfolder within the Models folder, and include some descriptive text document.
