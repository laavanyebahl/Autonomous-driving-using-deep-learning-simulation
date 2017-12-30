# Behaviorial Cloning Project
Used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

Simulator is used to steer a car around a track for data collection. Image data and steering angles were used to train a neural network and then use this model to drive the car autonomously around the track.

The project has five main files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car )
* model.h5 (a trained Keras model)
* a report 
* video.mp4 (a video recording of your vehicle driving autonomously around the track )


## Details About Files In This Directory

### `drive.py`

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

