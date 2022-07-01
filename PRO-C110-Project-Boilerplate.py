
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		#resize the frame
		img = cv2.resize(frame,(224,224))
		# expand the dimensions
		test_img = np.array(img,dtype = np.float32)
		test_img = np.expand_dims(test_img,axis = 0)
		# normalize it before feeding to the model
		normalise_img = test_img/255.2
		# get predictions from the model
		prediction = model.predict(normalise_img)
		print("Prediction = ",prediction)
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
