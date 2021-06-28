production = True
crop_value = 400   #In case nedd to change standar crop value
embryo_rad_value = 200 #In case nedd to change standar crop value


import argparse

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-i','--indir', action="store", dest='indir', help='Directory containing frames', default=False, required = True)
parser.add_argument('-o','--out', action="store", dest='out', help='Output', default=False, required = True)
parser.add_argument('-l','--loops', action="store", dest='loops', help='Loops', default=False, required = True)

args = parser.parse_args()

from scipy.stats import variation
import glob2
import cv2
import numpy as np
from skimage import color, feature
from skimage.util import img_as_ubyte, img_as_float
from collections import Counter,OrderedDict
import pathlib
from matplotlib import pyplot as plt
#if production == False:    
    #%matplotlib inline



indir = args.indir #"/nfs/research/birney/users/marcio/medaka_images/191122101610_BW7_28C_EtOH_Test3_R1/exp/croppedRAWTiff"
out_dir = args.out #"/nfs/research/birney/users/marcio/medaka_images/191122101610_BW7_28C_EtOH_Test3_R1/exp/croppedRAWTiff_2"
loops = args.loops

print(loops)
loops = loops.split('.')
print("loops")
print(loops)


loops = ["LO001"]



# start of functions

#################################################################

def normFrame(frame):
    
    #Convert RGB to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    norm_frame = np.uint8(frame / np.max(frames_appended) * 255)
    #Convert scaled greyscale back to RGB
    norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)       
    return(norm_frame)

###############################################################

def detectEmbryo(frame):

    #Find circle i.e. the embryo in the yolk sac 
    img_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Blur
    img_grey = cv2.GaussianBlur(img_grey, (9, 9), 0)
    #Edge detection
    edges = feature.canny(img_as_float(img_grey), sigma=3)
    edges = img_as_ubyte(edges)

    #Circle detection
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=30, minRadius = 150, maxRadius = 400)

    #If fails to detect embryo following edge detection, 
    #try with the original image 
    if circles is None:        
        circles = cv2.HoughCircles(img_grey, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=30, minRadius = 150, maxRadius = 400)
    
    if circles is None:       
        # Both trials failed to detect embryo, then try to threshold the image first, and try again              
        ret,th = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)          
        circles = cv2.HoughCircles(th, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=30, minRadius = 150, maxRadius = 400)
        
    if circles is not None:
        #Sort detected circles
        circles = sorted(circles[0],key=lambda x:x[2],reverse=True)
        #Only take largest circle to be embryo
        circle = np.uint16(np.around(circles[0]))
        #Circle coords
        centre_x = circle[0]
        centre_y = circle[1]
        radius = circle[2]
        
        
        x1 = centre_x - radius
        x2 = centre_x + radius
        y1 = centre_y - radius
        y2 = centre_y + radius
        
        #Round coords
        x1_test = 100 * round(x1 / 100)
        x2_test = 100 * round(x2 / 100)
        y1_test = 100 * round(y1 / 100)
        y2_test = 100 * round(y2 / 100)

        #If rounded figures are greater than x1 or y1, take 50 off it 
        if x1_test > x1:
            x1 = x1_test - 50 
        else:
            x1 = x1_test

        if y1_test > y1:
            y1 = y1_test - 50 
        else:
            y1 = y1_test

        #If rounded figures are less than x2 or y2, add 50 
        if x2_test < x2:
            x2 = x2_test + 50 
        else:
            x2 = x2_test

        if y2_test < y2:
            y2 = y2_test + 50 
        else:
            y2 = y2_test

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

    else:
        print('Fails all trials to detect embryo in detect_embryo()')
        
    return(circle, x1, y1, x2, y2)

#########################################################


# end of functions
for loop in loops:
    
    #these will be used to build the main plot, in which we will subplot the last cropped frame of each well
    axes = []
    rows = 8
    cols = 12
    fig=plt.figure(figsize=(10,28))
    suptitle = plt.suptitle('General view of every cropped well, loop: ' + loop, y=1.01, fontsize=14, color='blue')            
    
    for well_number in range(1,2):
        real_number = well_number  #this will be used to plot every subplot
        
        if well_number < 10:
            well_number = "WE0000" + str(well_number)
        else:
            well_number = "WE000" + str(well_number)
            
        well_frames = glob2.glob(indir + '/*' + well_number + '*.tif') + glob2.glob(indir + '/*' + well_number + '*.tiff')
        well_frames = [fname for fname in well_frames if loop in fname]
        frame_format = "tiff"
        if not well_frames:
            print("no tiff frames")
            well_frames = glob2.glob(indir + '/*' + well_number +'*.jpeg') + glob2.glob(indir + '/*' + well_number + '*.jpg')
            well_frames = [fname for fname in well_frames if loop in fname]
            frame_format = "jpg"
            if not well_frames:
                print('ERROR: no files detected. Is the path to the files correct or the well/loop exists?')
                if production == True:
                    exit()
                else:
                    print(fake_string_to_raise_an_error)

        print(well_number)            
            
        frames_appended = []
        counter = 0
        for frame in well_frames:
            frame_resulting = cv2.imread(frame,1)
            frames_appended.append(frame_resulting)
            frame_globalsize = frame_resulting.shape    # will be used as a general size for cropping wells where wrong circles was detected
            counter += 1
            print(counter)




        sizes = {}
        circle_x = {}
        circle_y = {}
        circle_radii = {}

        counter = 0
        normalyzed_frame_array = []
        for frame in frames_appended:
            #normalyzed_frame = normFrame(frame)
            #normalyzed_frame_array.append(normalyzed_frame)
            print(counter)

            if counter <= 5: 

                normalyzed_frame = normFrame(frame)
                circle, x1, y1, x2, y2 = detectEmbryo(normalyzed_frame)


                #crop_img = img[y1:y2, x1:x2]
                #crop_size = (x2 - x1) * (y2 - y1)
                #crop_id = (well_number, loop, crop_size)
                #crop_params[crop_id] = [x1, y1, x2, y2]

                try:   #append if exists or
                    #sizes[well_number].append(sizes)
                    circle_x[well_number].append(circle[0])
                    circle_y[well_number].append(circle[1])
                    circle_radii[well_number].append(circle[2])
                except:  #create new if does not exist
                    #sizes[well_number] = [crop_size]
                    circle_x[well_number] = [circle[0]]
                    circle_y[well_number] = [circle[1]]
                    circle_radii[well_number] = [circle[2]]
                
                
                # Draw the circle                
                #cv2.circle(normalyzed_frame,(circle[0],circle[1]),2,(255,225,0),20)                
                #cv2.circle(normalyzed_frame,(circle[0],circle[1]),circle[2],(0,255,0),3)                
                #plt.imshow(normalyzed_frame)
                #plt.show()

                

                counter += 1
                
        print(circle_x)
        
        if len(circle_x[well_number]) >= 5:
            print("higher than 5")
            for well_number in circle_x.keys():

                x_coord = circle_x[well_number]
                print("x_cood")
                print(x_coord)
                print(type(x_coord))
                print("std:")
                
                variation_value = variation(x_coord)
                
                if variation_value < 0.01:
                    
                    y_coord = circle_y[well_number]
                    radius = circle_radii[well_number]

                    x_counts = Counter(x_coord)
                    print(x_counts)
                    y_counts = Counter(y_coord)
                    print(y_counts)
                    rad_counts = Counter(radius)
                    print(rad_counts)

                    #Use most common circle coords for the embryo
                    embryo_x, _ = x_counts.most_common(1)[0]
                    embryo_y, _ = y_counts.most_common(1)[0]
                    embryo_rad, _ = rad_counts.most_common(1)[0]

                    print(embryo_x, embryo_y, embryo_rad)
                else:
                    print("high circle variance, set a value for cropping around the edges...")
                    embryo_x = int(frame_globalsize[0]/2)
                    embryo_y = int(frame_globalsize[1]/2)
                    embryo_rad = 200    
                    

            #circle[0] = embryo_x
            #circle[1] = embryo_y
            #circle[2] = embryo_rad

                # Draw the center of the circle
                #cv2.circle(img_out,(circle[0],circle[1]),2,(255,225,0),20)
                # Draw the circle
                #cv2.circle(img_out,(circle[0],circle[1]),circle[2],(0,255,0),3)



        else:
            Print("Error: it was not possible to detect the embryo in at least 5 frames, cropping around the edges...")
            embryo_x = int(frame_globalsize[0]/2)
            embryo_y = int(frame_globalsize[1]/2)
            embryo_rad = embryo_rad_value    
            
        path = pathlib.Path(out_dir)
        path = path.parent
        original_path = path #will be used to sabe results
        
        for count, img in enumerate(frames_appended):    
            path = pathlib.PurePath(well_frames[count])
            file_name = path.name
            
            #cv2.circle(img,(embryo_x,embryo_y),2,(255,225,0),20)
            #cv2.circle(img,(embryo_x,embryo_y),embryo_rad,(0,255,0),3)
            
            #cv2.circle(img,(embryo_x-embryo_rad,0),2,(255,0,255),50)
            #cv2.circle(img,(embryo_x+embryo_rad,0),2,(255,0,255),50)
            
            #cv2.circle(img,(0, embryo_y-embryo_rad),2,(255,0,255),50)
            #cv2.circle(img,(0, embryo_y+embryo_rad),2,(255,0,255),50)



            
            #plt.imshow(img) 
            #plt.show()
            
           
            cut_image = img[embryo_y-crop_value:embryo_y+crop_value, embryo_x-crop_value:embryo_x+crop_value]
            
            #plt.imshow(cut_image) # plot in panel the last cropped image from the loop above
            #plt.show()      
            
            
            
            cv2.imwrite(out_dir + "/" + file_name, cut_image)
            
        #plt.figure(figsize=(1, 1))   
        #create a subplot for the first frame. Other informations for the graph are above, at the beginning of the loop
        axes.append(fig.add_subplot(rows, cols, real_number))
        subplot_title=("well: "+str(well_number))
        axes[-1].set_title(subplot_title, fontsize=11, color='blue') 
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()      
        plt.imshow(cut_image) # plot in panel the last cropped image from the loop above
        #plt.show()

    #save the graph when the loop for every well finishes. Append tghe loop name to the graph filename            
    plt.savefig(str(original_path) + '/' + 'crop_overview_loop_' + loop + '.png', bbox_extra_artists=(suptitle,), bbox_inches="tight")