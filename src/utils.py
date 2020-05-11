import numpy as np 
import numpy as np 
import cv2
# uses https://gist.github.com/pebbie/c2cec958c248339c8537e0b4b90322da for skeletonization
from bwmorph_thin import bwmorph_thin as bwmorph
import os.path as path
import sys
from keras.engine.topology import Layer
import keras.backend as K


def test_img(model,img,patch_size=(256,256,3),stride=(10,10)):
    a,b,_=img.shape
    if(min(img.shape)<256):
        print("image size is <256")
        return(False,-1,-1)
    idy=np.r_[0:a-patch_size[0]:stride[0]]
    idx=np.r_[0:b-patch_size[1]:stride[1]]
    idy=np.append(idy,a-patch_size[0])
    idx=np.append(idx,b-patch_size[1])
    L=[]
    for dx in idx:
        for dy in idy:
            patch=img[dy:dy+patch_size[0],dx:dx+patch_size[1],:]
            L.append(patch)
            

    L=np.array(L)   
    #L=np.expand_dims(L,-1)

    L_out=model.predict(L)
    #out_img=np.zeros(img.shape,dtype='float32')
    out_img=np.zeros((a,b),dtype='float32')
    c=0
    for dx in idx:
        for dy in idy:
            #out_img[dy:dy+patch_size[0],dx:dx+patch_size[1]]=L_out[c]
            out_img[dy:dy+patch_size[0],dx:dx+patch_size[1]]=L_out[c,:,:,0]
            c=c+1
       

    return out_img,L,L_out



def drd_fn(im, im_gt):
	height, width = im.shape
	neg = np.zeros(im.shape)
	neg[im_gt!=im] = 1
	y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
	
	n = 2
	m = n*2+1
	W = np.zeros((m,m), dtype=np.uint8)
	W[n,n] = 1.
	W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	W[n,n] = 1.
	W = 1./W
	W[n,n] = 0.
	W /= W.sum()
	
	nubn = 0.
	block_size = 8
	for y1 in xrange(0, height, block_size):
		for x1 in xrange(0, width, block_size):
			y2 = min(y1+block_size-1,height-1)
			x2 = min(x1+block_size-1,width-1)
			block_dim = (x2-x1+1)*(y1-y1+1)
			block = 1-im_gt[y1:y2, x1:x2]
			block_sum = np.sum(block)
			if block_sum>0 and block_sum<block_dim:
				nubn += 1

	drd_sum= 0.
	tmp = np.zeros(W.shape)
	for i in xrange(min(1,len(y))):
		tmp[:,:] = 0 

		x1 = max(0, x[i]-n)
		y1 = max(0, y[i]-n)
		x2 = min(width-1, x[i]+n)
		y2 = min(height-1, y[i]+n)

		yy1 = y1-y[i]+n
		yy2 = y2-y[i]+n
		xx1 = x1-x[i]+n
		xx2 = x2-x[i]+n

		tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
		tmp *= W

		drd_sum += np.sum(tmp)
	return drd_sum/nubn


def metric_cal(inp_path,gt_path):
    im = cv2.imread(inp_path,0)
    im_gt = cv2.imread(gt_path, 0)

    height, width = im.shape
    npixel = height*width

    im[im>0] = 1
    gt_mask = im_gt==0
    im_gt[im_gt>0] = 1

    sk = bwmorph(1-im_gt)
    im_sk = np.ones(im_gt.shape)
    im_sk[sk] = 0

    kernel = np.ones((3,3), dtype=np.uint8)
    im_dil = cv2.erode(im_gt, kernel)
    im_gtb = im_gt-im_dil
    im_gtbd = cv2.distanceTransform(1-im_gtb, cv2.DIST_L2, 3)

    nd = im_gtbd.sum()

    ptp = np.zeros(im_gt.shape)
    ptp[(im==0) & (im_sk==0)] = 1
    numptp = ptp.sum()

    tp = np.zeros(im_gt.shape)
    tp[(im==0) & (im_gt==0)] = 1
    numtp = tp.sum()

    tn = np.zeros(im_gt.shape)
    tn[(im==1) & (im_gt==1)] = 1
    numtn = tn.sum()

    fp = np.zeros(im_gt.shape)
    fp[(im==0) & (im_gt==1)] = 1
    numfp = fp.sum()

    fn = np.zeros(im_gt.shape)
    fn[(im==1) & (im_gt==0)] = 1
    numfn = fn.sum()

    precision = numtp / (numtp + numfp)
    recall = numtp / (numtp + numfn)
    precall = numptp / np.sum(1-im_sk)
    fmeasure = (2*recall*precision)/(recall+precision)
    pfmeasure = (2*precall*precision)/(precall+precision)

    mse = (numfp+numfn)/npixel
    psnr = 10.*np.log10(1./mse)

    nrfn = numfn / (numfn + numtp)
    nrfp = numfp / (numfp + numtn)
    nrm = (nrfn + nrfp)/2

    im_dn = im_gtbd.copy()
    im_dn[fn==0] = 0
    dn = np.sum(im_dn)
    mpfn = dn / nd

    im_dp = im_gtbd.copy()
    im_dp[fp==0] = 0;
    dp = np.sum(im_dp)
    mpfp = dp / nd

    mpm = (mpfp + mpfn) / 2
    drd = drd_fn(im, im_gt)
    print "F-measure\t: {0}\npF-measure\t: {1}\nPSNR\t\t: {2}\nNRM\t\t: {3}\nMPM\t\t: {4}\nDRD\t\t: {5}".format(fmeasure, pfmeasure, psnr, nrm, mpm, drd)
    return (fmeasure, pfmeasure, psnr, nrm, mpm, drd)
	

#metric_cal(sys.argv[1],sys.argv[2])



























class AGGLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AGGLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z3 = inputs[0]    #prob of image size
        z4 = inputs[1]	 #background color
        I = inputs[2]

        z4_new=K.expand_dims(z4,axis=1)
        z4_new=K.expand_dims(z4_new,axis=1)
        z4_new=K.tile(z4_new,[1,256,256,1])


        z3_temp=K.repeat_elements(z3,rep=3,axis=-1)
	#z3_temp=z3


        z3_new=(1-z3_temp)*z4_new+z3_temp*I
      
	z3_new=K.clip(z3_new,0,1)
        self.out_shape=z3_new.get_shape()
        return z3_new

    def compute_output_shape(self, input_shape):

        #return (a.shape[0],a.shape[1],a.shape[2],3)
        #return(5,256,256,3)
        return self.out_shape

class AGGLayer_DIBCO(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AGGLayer_DIBCO, self).__init__(**kwargs)

    def call(self, inputs):
        z3 = inputs[0]			#prob of image size
        z4 = inputs[1]			#background color

        z4_new=K.expand_dims(z4,axis=1)
        z4_new=K.expand_dims(z4_new,axis=1)
        z4_new=K.tile(z4_new,[1,256,256,1])

	z4_new=K.mean(z4_new,axis=-1,keepdims=True)

        #z3_temp=z3


        z3_new=(1-z3)*z4_new+z3*(1-z4_new)


        z3_new=K.clip(z3_new,0,1)
        self.out_shape=z3_new.get_shape()
 
	loss=K.mean(K.abs(1-z4_new))
        self.add_loss(loss, inputs=inputs)


        return z3_new

    def compute_output_shape(self, input_shape):

        #return (a.shape[0],a.shape[1],a.shape[2],3)
        #return(5,256,256,3)
        return self.out_shape



















