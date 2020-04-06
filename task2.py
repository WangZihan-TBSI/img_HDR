# HDR image creation
# Coder: Zihan Wang， Chuhui Wang

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 

#get image directory & exposure time
def getImageProfile():
    f = open('./images/list.txt', 'r')
    img_folder = './images/'
    data_lists = f.readlines()
    ET = []
    images_dir = []
    for data in data_lists:
        data1 = data.strip('\n')
        data2 = data1.split(' ')
        # record the image directory
        images_dir.append(os.path.join(img_folder, data2[0]))
        # Compute the exposure times in seconds
        ET.append(1 / np.array(data2[1], dtype=np.float32))
    ET = np.array(ET, dtype=np.float32)
    return images_dir, ET

# compute the average of images    
def average(images_dir):
    total_img = np.zeros(np.shape(cv2.imread(images_dir[0])), float)
    for image_file in images_dir:
        total_img += cv2.imread(image_file).astype(np.float)
    average_img = total_img/len(images_dir)
    average_img = cv2.normalize(average_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('average_img.png', average_img)
# 

# calculate response function
def filmResponse(images, ET,weights,ln_dt):
    num_frames = len(ET)
    num_samples = round(255/(num_frames-1)*2)
    img_pixels = np.shape(images[0])[1]*np.shape(images[0])[0]
    step = img_pixels/num_samples
    sample_indices = np.array(range(1, img_pixels, int(step)))
    sample_indices_x = np.floor_divide(sample_indices, np.shape(images)[2])
    sample_indices_y = np.mod(sample_indices, np.shape(images)[2])
    # Preallocate space for results in the order of red/green/blue.
    z = np.zeros([num_samples, num_frames, 3], dtype=np.float32)
    # sample the images
    for i in range(num_frames):
        red_image = images[i,:,:,0]
        for j in range(num_samples):
            z[j, i, 0] = red_image[sample_indices_x[j]][sample_indices_y[j]]
            green_image=images[i,:, :,1]
            z[j, i, 1] = green_image[sample_indices_x[j]][sample_indices_y[j]]
            blue_image=images[i,:, :, 2]
            z[j, i, 2] = blue_image[sample_indices_x[j]][sample_indices_y[j]]
    
    smoothing_factor = 50
    n = 256
    A = np.zeros((num_samples*num_frames+n+1,n+num_samples))
    b = np.zeros((num_samples*num_frames+n+1))
    response = np.zeros([n,1,3],dtype=np.float32)
    # data-fitting equation
    for color_channel in range(3):
        k = 0
        for i in range(num_samples):
            for j in range(num_frames):
                A[k, int(z[i, j, color_channel])] = weights[int(z[i, j, color_channel])]
                A[k, n+i] = -weights[int(z[i, j, color_channel])]
                b[k] = weights[int(z[i, j, color_channel])]*ln_dt[j]
                k = k+1
            # fix the curve by setting middle value to 0    
        A[k,129] = 1
        k = k+1
        # include smoothness equation
        for i in range(n-1):
            A[k, i] = smoothing_factor*weights[i+1]
            A[k, 1+i] = -2*smoothing_factor*weights[i+1]
            A[k, 2+i] = smoothing_factor*weights[i+1]
            #这里有bug
        x = np.dot(np.linalg.pinv(A),b)
        response[:,0,color_channel]=x[:n]
    return response, weights
def my_mapping(subject,reference):
    [width, height, num_channels) = np.shape(subject)
    result = np.zeros([width, height, num_channels])
    for x in range(width):
            for y in range(height):
                for z in range(num_channels):
                    result[x, y, z] = reference[subject[x, y, z]]
    return result
# my own calibration function
def OwnMerge(images, ET, response,weights,ln_dt):
    [num_exposures, width, height, num_channels] = np.shape(images)
    numerator = np.zeros([height, width, num_channels])
    denominator = np.zeros([height, width, num_channels])
    curr_num = np.zeros([height, width, num_channels])
    for i in range(num_exposures):
        curr_image = images[i,:,:,:]
        curr_red = curr_image[:,:,0]
        curr_green = curr_image[:,:,1]
        curr_blue = curr_image[:,:,2]
        curr_weight = my_mapping(curr_image,weights)
        curr_num[: , : , 1] = dot(curr_weight[: , : , 1], (my_mapping(curr_red,g_red)- ln_dt(i)))
        curr_num[:, :, 2] = dot(curr_weight[:, :, 2],
                                (my_mapping(curr_green, g_green) - ln_dt(i)))
        curr_num[:, :, 3] = dot(curr_weight[:, :, 3],
                                (my_mapping(curr_blue, g_blue) - ln_dt(i)))
        

    return merge

# generate HDR image
def HDR(files, ET):
   # Read all images with OpenCV
    images = list([cv2.imread(f) for f in files])
    images = np.array(images)
    print('images shape', np.shape(images))
    ln_dt = np.log(ET)
    weights = [min(i, 256-i) for i in range(1, 257)]
    # Compute the response curve
    # D : optical density
    # X : Exposure
    # E : irradiance
    # ET : exposure time
    [response, weights] = filmResponse(images, ET,weights,ln_dt)
 
    # TODO: 自己写一下createCalibrateDebevec()
    # calibration = cv2.createCalibrateDebevec() 
    # response = calibration.process(images, ET)
    print('response shape',np.shape(response))
    fig = plt.figure()
    fig.suptitle('Response Curve')
    print(np.shape(response))
    r = response[:,:,0]
    g = response[:,:,1]
    b = response[:,:,2]
    x = range(256)
    plt.plot(x, r, color='red')
    plt.plot(x, g, color='green')
    plt.plot(x, b, color='blue')
    plt.xlabel("Pixel Value")
    plt.ylabel("Pixel Exposure(log)")
    # plt.show()
    plt.savefig('response_curve.png')
    # Compute the HDR image
    # TODO: 自己写一下createMergeDebevec()
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, ET, response)
    cv2.imwrite('hdr_image.hdr', hdr)
    # tone mapping
    durand = cv2.createTonemapDurand(gamma=3)
    ldr = durand.process(hdr)
    # Tonemap operators create floating point images with values in the 0..1 range
    # This is why we multiply the image with 255 before saving
    cv2.imwrite('durand_image.png', ldr * 255)

def main():
    [images_dir, ET] = getImageProfile()
    average(images_dir)
    HDR(images_dir, ET)
# execute from main function
if __name__ == '__main__':
    main()
