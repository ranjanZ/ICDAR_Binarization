from train_model1 import model_DIBCO,model_ISI
from utils import *
import os
from scipy import misc
import 





INP_PATH_DIR=sys.argv[1]
out_DIR=sys.argv[2]




M=[]

for d in os.listdir(INP_PATH_DIR):
    inp_file_path=INP_PATH_DIR+d
    img=misc.imread(inp_file_path,mode="RGB")/255.0

    model=model_DIBCO()
    model.load_weights("./models/model_weights_DIBCO.h5")
    out_img,_,_=test_img(model,img,patch_size=(256,256,3),stride=(100,100))
	 
    misc.imsave(OUT_DIR+d,out_img);







