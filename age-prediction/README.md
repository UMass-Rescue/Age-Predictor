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

To use this repo without modifying configuration.json file create images, cropped-faces and output-files directory in age-prediction folder.

In linux based system

```
mkdir images cropped-faces output-files
```

## Running the age predictor
There are three ways the age predictor can be executed.

Below steps assume that all the preparations from the "getting started" part has been done.

### 1. Crop faces from input image and run age predictor
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

### 2. Run age predictor on already cropped faces

If cropped faces are already available then age predictor can directly be run on them. 

Save cropped faces into a directory and ensure that configuration.json file has the correct name.

To run age predictor, execute below command
```
python3 predict-age-on-cropped-faces.py
```
Above script reads images from cropped faces directory, does age-prediction and writes the output into appropriate file in output-files directory.

### 3. Run age predictor on a single image
This returns number of faces in input image, bounding box and age for each face.

To run age predictor, execute below command
```
python3 predict-age-single-image.py -i <input-image-path>
```
where <image-path> is the absolute path of image or name of image if it is located in same directory as this script
  
## Testing the age predictor
Test program takes cropped faces as input. These images are placed in four directories representing four age classes "below-9", "9-13", "14-17" and "18-above". If cropped faces are not available a script is provided to crop the faces.
Go to directory "test-scripts"
```
cd test-scripts
```
### 1. Crop faces
This can be used for cropping the faces from images if not already available.
configuration.json file contains the paths for input images and cropped faces.

Key "input-images-directory" stores the path of input images(to crop faces from).

Key "cropped-faces-directory" stores the path of cropped faces. Program will save the cropped faces in this directory.

Configuration file can be modified to change the paths. To use the program as it is create directories images and cropped-faces.
```
mkdir images cropped-faces
```
Save input images to images directory and execute "crop-faces-test.py".
```
python3 crop-faces-test.py
```
Crpped faces will be stored in cropped-faces directory.

### 2. Test age predictor
Create a root directory to store class labeled input images. Root directory name is "test-images" currently but it can be modified. Instructions are provided in ipython notebook itself.
Open "predict-age-gen-metric.ipynb" in jupyter notebook.
```
jupyter notebook predict-age-gen-metric.ipynb
```
Instruction to modify the root directory path is provided in notebook. Create four directories representing four age classes "below-9", "9-13", "14-17" and "18-above" in root directory and save the corresponding face images here. Execute all the cells in ipython notebook. Accuracy, precision and recall will be displayed in last cell.

## Performance bench-mark
I have created a custom dataset to test this model. Link to dataset is: https://umass.box.com/s/rg4i07jxkvfky8s0nade2o5re33pfg97

Precision:: "Below-9":0.373,	"9-13":0.365,	"14-17":0.402,	"18-above":0.840

Recall::	"Below-9":0.339,	"9-13":0.642,	"14-17":0.734,	"18-above":0.382


## Acknowledgments

* https://github.com/Raschka-research-group/coral-cnn/
* Consistent Rank Logits for Ordinal Regression with Convolutional Neural Networks https://arxiv.org/abs/1901.07884
