# Age-Estimation

## Getting Started

Following are the instructions to download and install the dependencies to successfully run the project.

### Prerequisites
Ensure to download the Python 3.7 version from https://www.anaconda.com/distribution/, and follow the installation procedure.

### Installing

Create a conda virtual environment with python 3.7

```
conda create -n age-estimator python=3.7
```

Activate the virtual environment

```
conda activate age-estimator
```

Install pip and pytorch through conda

```
conda install pip
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
Clone this repository to local machine and go to folder age-prediction.

Install requirement.txt using pip. This will install all the dependencies for project.

```
pip install -r requirements.txt
```
### Some more preparations

Create model directory in age-prediction folder.

In linux based system

```
mkdir model
```
Download pretrained model from <a href="https://drive.google.com/open?id=1Lo3CDSXLMg68lsLNkC7a523ScA6A-TBs">here</a>
and save into "model" folder

configuration.json file contains the paths for input images, cropped faces and output files.

Key "input-images-directory" stores the path of input images. Images from this directory will be input to face-detector.

Key "cropped-faces-directory" stores the path of cropped faces. Faces from input images will be cropped and stored to this directory.
These cropped faces will be input to age predictor.

Key "output-files-directory" stores the output files. Output files are named as per age class and will list the input images that fit the age class.  
## Running the project

To use this repo without modifying configuration.json file create cropped-faces and output-files directory in age-prediction folder.

In linux based system

```
mkdir cropped-faces output-files
```

## Running the age predictor
There are two ways the age predictor can be executed.

Below steps assume that all the preparations from the "getting started" part has been done.

### Crop faces from input image and run age predictor
If an image contains multiple faces, the faces will be cropped and age prection will run on all the faces in the image. It will classify the
input image as per the minimum age found in image. Example: If an image contains two faces aged 8 and 18 years old. Then image will be classified in
"below 9" class.

To crop the faces, execute below command
```
python3 crop-faces.py
```
Above scripy reads images files from input image directory and saves the cropped faces into cropped faces directory.

To run age predictor, execute below command
```
python3 predict-age.py
```
Above script reads images from cropped faces directory, does age-prediction and writes the output into appropriate file in output-files directory.

### Run age predictor on already cropped faces

If cropped faces are already available then age predictor can directly be run on them. 

Save cropped faces into a directory and ensure that configuration.json file has the correct name.

To run age predictor, execute below command
```
python3 predict-age-on-cropped-faces.py
```
Above script reads images from cropped faces directory, does age-prediction and writes the output into appropriate file in output-files directory.

## Acknowledgments

* https://github.com/Raschka-research-group/coral-cnn/compare
* Consistent Rank Logits for Ordinal Regression with Convolutional Neural Networks https://arxiv.org/abs/1901.07884
