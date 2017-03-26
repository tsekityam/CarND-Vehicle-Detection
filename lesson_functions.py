import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.feature import hog
from model_parameters import *

# Define a function to return an RGB image that has been converted to given color space
def get_feature_image(image, color_space='RGB'):
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)

    return feature_image

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, spatial_color_space='RGB', hist_color_space='RGB', hog_color_space='RGB',
                     spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion

        if spatial_feat == True:
            spatial_feature_image = get_feature_image(image, spatial_color_space)
            spatial_features = bin_spatial(spatial_feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_feature_image = get_feature_image(image, hist_color_space)
            hist_features = color_hist(hist_feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            hog_feature_image = get_feature_image(image, hog_color_space)
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(hog_feature_image.shape[2]):
                    hog_features.append(get_hog_features(hog_feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(hog_feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              spatial_color_space='RGB', hist_color_space='RGB', hog_color_space='RGB', hog_channel=0,
              spatial_feat=True, hist_feat=True, hog_feat=True):

    bbox_list = []
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    spatial_ctrans_tosearch = get_feature_image(img_tosearch, color_space=spatial_color_space)
    hist_ctrans_tosearch = get_feature_image(img_tosearch, color_space=hist_color_space)
    hog_ctrans_tosearch = get_feature_image(img_tosearch, color_space=hog_color_space)

    for scale in scales:
        if scale != 1:
            imshape = img_tosearch.shape
            spatial_ctrans_tosearch = cv2.resize(spatial_ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            hist_ctrans_tosearch = cv2.resize(hist_ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            hog_ctrans_tosearch = cv2.resize(hog_ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = hog_ctrans_tosearch[:,:,0]
        ch2 = hog_ctrans_tosearch[:,:,1]
        ch3 = hog_ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = 1

        if hog_feat == True:
            # Compute individual channel HOG features for the entire image
            if hog_channel == 'ALL':
                hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
            elif hog_channel == 0:
                hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            elif hog_channel == 1:
                hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            elif hog_channel == 2:
                hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                file_features = []

                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Get color features
                if spatial_feat == True:
                    # Extract the image patch
                    spatial_subimg = cv2.resize(spatial_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                    spatial_features = bin_spatial(spatial_subimg, size=spatial_size)
                    file_features.append(spatial_features)
                if hist_feat == True:
                    # Extract the image patch
                    hist_subimg = cv2.resize(hist_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                    hist_features = color_hist(hist_subimg, nbins=hist_bins)
                    file_features.append(hist_features)
                if hog_feat == True:
                    # Extract HOG for this patch
                    if hog_channel == 'ALL':
                        hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    elif hog_channel == 0:
                        hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_feat1))
                    elif hog_channel == 1:
                        hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_feat2))
                    elif hog_channel == 2:
                        hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_feat3))
                    file_features.append(hog_features)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack(np.array(file_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

max_stack_size = 5
def get_image_with_car_highlighted(image, svc, X_scaler, matched_windows_stack):

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    scales = np.linspace(1.0, 2.5, num=7)
    bbox_list = find_cars(image, y_start_stop[0], y_start_stop[1], scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                          spatial_color_space=spatial_color_space, hist_color_space=hist_color_space, hog_color_space=hog_color_space, hog_channel=hog_channel,
              spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    matched_windows_stack.append(bbox_list)
    # ensure that the stack size is within the range of allowance
    if len(matched_windows_stack) > max_stack_size:
      matched_windows_stack = matched_windows_stack[-max_stack_size:]

    combined_bbox_list = []
    for matched_windows in matched_windows_stack:
        combined_bbox_list = combined_bbox_list + matched_windows

    # Add heat to each box in box list
    heat = add_heat(heat,combined_bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,7*len(matched_windows_stack))

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

def get_image_with_matched_window_highlighted(image, svc, X_scaler):

    scales = np.linspace(1.0, 2.5, num=7)
    bbox_list = find_cars(image, y_start_stop[0], y_start_stop[1], scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                          spatial_color_space=spatial_color_space, hist_color_space=hist_color_space, hog_color_space=hog_color_space, hog_channel=hog_channel,
                          spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    draw_img = draw_boxes(np.copy(image), bbox_list)

    return draw_img
