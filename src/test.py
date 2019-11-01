from train_model1 import model_DIBCO,model_ISI
from utils import *
import os
from scipy import misc
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import 



#INP_PATH_DIR="../dataset/ISI_letter/train/X/"
#GT_DIR="../dataset/ISI_letter/train/Y/"
#INP_PATH_DIR="../dataset/DIBCO/train/X/"
#GT_DIR="../dataset/DIBCO/train/Y/"
#INP_PATH_DIR="../dataset/DIBCO/test/2017/DIBCO_2017/"
#GT_DIR="../dataset/DIBCO/test/2017/DIBCO_2017_GT/"
#INP_PATH_DIR="../dataset/DIBCO/test/2018/DIBCO_2018/"
#GT_DIR="../dataset/DIBCO/test/2018/DIBCO_2018_GT/"


INP_PATH_DIR=sys.argv[1]
GT_DIR=sys.argv[2]




M=[]

for d in os.listdir(INP_PATH_DIR):
    inp_file_path=INP_PATH_DIR+d
    gt_file_path=GT_DIR+d[:-4]+"_gt."+d[-3:]
    #change is file names are different
    gt_file_path=GT_DIR+d
    
    img=misc.imread(inp_file_path,mode="RGB")/255.0
    img_gt=misc.imread(gt_file_path,"L")/255.0


    model=model_DIBCO()
    model.load_weights("./models/model_weights_DIBCO.h5")
    #model=model_ISI()
    #model.load_weights("./models/model_weights_isi.h5")
    out_img,_,_=test_img(model,img,patch_size=(256,256,3),stride=(100,100))
	 
    ssim_m=ssim(out_img,img_gt)
    psnr_m=psnr(out_img,img_gt)
    print ssim_m,psnr_m
	 
    misc.imsave("../dataset/out_2017/"+d,out_img);
    #m=metric_cal("../dataset/out_2017/"+d,gt_file_path)
    #M.append(m)
    #print m







