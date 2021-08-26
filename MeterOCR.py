# import the necessary packages
import numpy as np
import time
import cv2
import os
import argparse
configPath_for_feature = "models/box.cfg"
weightsPath_for_feature = "models/box.weights"
configPath_for_digits = "models/meter.cfg"
weightsPath_for_digits = "models/meter.weights"
feature_net = cv2.dnn_Net
digit_net = cv2.dnn_Net
img_dir = "./util_images_dev"

def load_model():
	global feature_net, digit_net
	print("[INFO] loading model from disk...")
	feature_net = cv2.dnn.readNetFromDarknet(configPath_for_feature, weightsPath_for_feature)
	digit_net = cv2.dnn.readNetFromDarknet(configPath_for_digits, weightsPath_for_digits)
def detect_feature(image):

	(H, W) = image.shape[:2]
	# determine only the *output* layer names that we need from YOLO
	ln = feature_net.getLayerNames()
	ln = [ln[i[0] - 1] for i in feature_net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (300, 200), swapRB=True, crop=False)
	feature_net.setInput(blob)
	start = time.time()
	layerOutputs = feature_net.forward(ln)
	end = time.time()

	# show timing information
	print("[INFO] Detecting the featuer took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	CONFIDENCE = 0.5
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence >= CONFIDENCE:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				res_img = cv2.rectangle(image, (x, y), (x+width, y+height), (0,0,255), 4)
		# get the feature image and rotat it with 270 degree
		cro_img = image[y :y + height , x:x + width ]
		rotate_img = cv2.rotate(cro_img, rotateCode = 2)
		return rotate_img

def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', type=str, default='./util_images_dev', help='image file')
	opt = parser.parse_args()
	return opt

def main(source):

	mode = 0
	if source != img_dir:
		source, file_name = os.path.split(os.path.abspath(source))
		mode = 1
	f_lists = os.listdir(source)
	fo = open("results/result.txt", "w")

	for f_name in f_lists:

		if mode == 1: f_name = file_name
		f_path = img_dir+"/"+f_name

		# load our input image and grab its spatial dimensions
		image = cv2.imread(f_path)
		image = cv2.resize(image, (1280, 985))
		image = detect_feature(image)
		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = digit_net.getLayerNames()
		ln = [ln[i[0] - 1] for i in digit_net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (250, 100), swapRB=True, crop=False)
		digit_net.setInput(blob)
		start = time.time()
		layerOutputs = digit_net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] Detecting the digits took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []
		rectXs = []
		digits = []
		sortedXs = []
		CONFIDENCE = 0.5
		# loop over each of the layer outputs

		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence >= 0.5:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					rectXs.append(x)
					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
					res_img = cv2.rectangle(image, (x, y), (x+width, y+height), (0,0,255), 4)
					cv2.putText(res_img, str(classID), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

		cv2.imwrite('results/' + f_name, res_img)
		for i in rectXs:
			sortedXs.append(i)
		sortedXs.sort()
		ptr = ''
		for i in sortedXs:
			k = rectXs.index(i)
			digits.append(classIDs[k])
			ptr += str(classIDs[k])
		ptr += '\n'
		print(f_name + ":" + ptr)
		fo.write(f_name + ":" + ptr)

		if mode == 1: break

if __name__ == '__main__':
	opt = parse_opt()
	load_model()
	main(**vars(opt))

