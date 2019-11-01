Code of the paper  Ranjan Mondal, Deepayan Chakraborty and Bhabatosh Chanda "Learning 2D Morphological Network for Old Document Image Binarization"

TODO: 

# Code of the paper "Morphological Networks for Image De-raining"

| Input        | De-Rained      | 
|:-------------:|:-------------:|
| ![input](https://dmtyylqvwgyxw.cloudfront.net/instances/132/uploads/images/photo/image/57150/large_871b49bd-3580-4ba1-a4a4-ac7842fb64ee.?v=1562226416)|






| ![input](https://raw.githubusercontent.com/ranjanZ/2D-Morphological-Network/master/data/input_images/52_in.png)| ![De-Rained](https://raw.githubusercontent.com/ranjanZ/2D-Morphological-Network/master/data/output/52_4small.png) |

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
This runs the code in the supplied images.
```
$python run.py ./data/input_images/ ./data/output/

```

## Files
```

.
├── bwmorph_thin.py
├── Dssim.py
├── generator.py
├── morph_layers2D.py
├── run.py
├── test.py
├── train_model1.py
├── train_model.py
└── utils.py

```

## Publication
Ranjan Mondal, Deepayan Chakraborty and Bhabatosh Chanda. "Learning 2D Morphological Network for Old Document Image Binarization" International Conference on Document Analysis and Recognition, 2019

Ranjan Mondal,Pulak Purkiat, Sanchayan Santra and Bhabatosh Chanda. "Morphological Networks for Image De-raining" Discrete Geometry for Computer Imagery, 2019


#If you are using this code please cite the paper


