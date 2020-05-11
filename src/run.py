from train_model import model_DIBCO,model_ISI
from utils import *
import os
from scipy import misc





INP_PATH_DIR=sys.argv[1]
out_DIR=sys.argv[2]




M=[]

for d in os.listdir(INP_PATH_DIR):
    inp_file_path=INP_PATH_DIR+d
    img=misc.imread(inp_file_path,mode="RGB")/255.0

    #model=model_DIBCO()
    #model.summary()
    #model.load_weights("./models/model_weights_DIBCO.h5")
    model=model_ISI()
    model.summary()
    model.load_weights("./models/model_weights_isi.h5")
    out_img,_,_=test_img(model,img,patch_size=(256,256,3),stride=(100,100))
	 
    misc.imsave(OUT_DIR+d,out_img);







