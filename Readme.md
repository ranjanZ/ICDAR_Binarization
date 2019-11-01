

#Code of the paper "Learning 2D Morphological Network for Old Document Image Binarization"
| Input   and  Output    | 
|:--------------------------:|
| ![input](https://dmtyylqvwgyxw.cloudfront.net/instances/132/uploads/images/photo/image/57150/large_871b49bd-3580-4ba1-a4a4-ac7842fb64ee.?v=1562226416)|







## Dependency
* For Running
    * Python2
    * keras(tensorflow-backend with channel last)
    * scipy
    * numpy
    * scikit-image
    * matplotlib

## Running
```
$ cd src/
$ python test.py  <old_binary_image_dir>   <ground_truth_dir>
```
This runs the code in the supplied images	 and ground truth images.
```
$python run.py ./data/input_images/ ./data/output/
```
This  generates the output only 
```
We encourange to train the model from  start by using the file train_model.py 
Set the data path in generator.py
call the function train_model()
```

## Files
```
├── bwmorph_thin.py
├── Dssim.py
├── generator.py
├── images
│   ├── input
│   └── output
├── models	#all the trained models
│   ├── model_weights1.h5
│   ├── model_weights_DIBCO.h5
│   ├── model_weights.h5
│   ├── model_weights_isi1.h5
│   └── model_weights_isi.h5
├── morph_layers2D.py			#defined morphologcal layers
├── run.py				
├── test.py				#to test the models,gven input and ground truth
├── train_model.py			#to tain the model 
└── utils.py

```

## Publication
Ranjan Mondal, Deepayan Chakraborty and Bhabatosh Chanda. "Learning 2D Morphological Network for Old Document Image Binarization" International Conference on Document Analysis and Recognition, 2019

Ranjan Mondal,Pulak Purkiat, Sanchayan Santra and Bhabatosh Chanda. "Morphological Networks for Image De-raining" Discrete Geometry for Computer Imagery, 2019


#If you are using this code please cite the paper


