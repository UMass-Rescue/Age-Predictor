from torch_mtcnn import detect_faces
from PIL import Image
from os import listdir
from os.path import isfile, join
import json
import csv
with open('configuration.json', 'r') as configfile:
    config_info=json.load(configfile)
print(config_info['input-images-directory'])
input_images_path=config_info['input-images-directory']
print(input_images_path)
image_files = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
cropped_images_path=config_info['cropped-faces-directory']
for file in image_files:
	image = Image.open(input_images_path+file).convert('RGB')
	bounding_boxes, landmarks = detect_faces(image)
	for i in range(len(bounding_boxes)):
		area = (bounding_boxes[i][0],bounding_boxes[i][1],bounding_boxes[i][2],bounding_boxes[i][3])
		t_image=image.crop(area)
		j=i+1
		t_image=t_image.save(cropped_images_path+str(j)+'-'+file)
            