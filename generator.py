import cv2
import numpy as np
import pandas as pd

class Generator:
    def __init__(self,input_shape,data_path,batch_size=250)
        self.new_size_col,self.new_size_row,self.ch = input_shape
        driveLog_fname = "{}/driving_log.csv".format(data_path)
        image_fdir = dirpath
        self.batch_size = batch_size
        self._threshold = 1
        drivelog = pd.read_csv(driveLog_fname,names=["center","left","right","steering","throttle","brake","speed"])

    @property    
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold_setter(self,x):
        self._threshold = x
        
    def augment_brightness_camera_images(self,image):
        image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        #print(random_bright)
        image1[:,:,2] = image1[:,:,2]*random_bright
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return image1

    def trans_image(self,image,steer,trans_range):
        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        steer_ang = steer + tr_x/trans_range*2*.2
        tr_y = 40*np.random.uniform()-40/2
        #tr_y = 0
        rows,cols,ch=image.shape
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

        return image_tr,steer_ang

    def preprocessImage(self,image):
        shape = image.shape
        # note: numpy arrays are (row, col)!
        image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
        image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
        #image = image/255.-.5
        return image

    def preprocess_image_file_train(self,line_data):
        i_lrc = np.random.randint(3)
        if (i_lrc == 0):
            path_file = line_data['left'][0].strip()
            shift_ang = .25
        if (i_lrc == 1):
            path_file = line_data['center'][0].strip()
            shift_ang = 0.
        if (i_lrc == 2):
            path_file = line_data['right'][0].strip()
            shift_ang = -.25
        y_steer = line_data['steering'][0] + shift_ang
        image = cv2.imread(path_file)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image,y_steer= self.trans_image(image,y_steer,100)

        image = self.augment_brightness_camera_images(image)

        image = self.preprocessImage(image)

        image = np.array(image)

        ind_flip = np.random.randint(2)
        if ind_flip==0:
            image = cv2.flip(image,1)
            y_steer = -y_steer

        return image,y_steer

    def generate_data(self,data,threshold):
        batch_size = self.batch_size
        batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
        batch_steering = np.zeros(batch_size)
        while 1:
            #batch_images=[]
            for i_batch in range(batch_size):
                i_line = np.random.randint(len(data))
                line_data = data.iloc[[i_line]].reset_index()

                keep_pr = 0
                #x,y = preprocess_image_file_train(line_data)
                while keep_pr == 0:
                    x,y = self.preprocess_image_file_train(line_data)

                    pr_unif = np.random
                    if abs(y)<.1:
                        pr_val = np.random.uniform()
                        if pr_val>self._threshold:
                            keep_pr = 1
                    else:
                        keep_pr = 1

                #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
                #y = np.array([[y]])
                #batch_images.append(x)
                batch_images[i_batch] = x
                batch_steering[i_batch] = y

            yield batch_images, batch_steering