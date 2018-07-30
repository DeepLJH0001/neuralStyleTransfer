# neuralStyleTransfer
A neural style transfer with keras
Uses a pre-trained model : VGG16
[Doc about VGG16] (Very Deep Convolutional Networks for Large-Scale Image Recognition)


# Built with
* [Keras with TensorFlow](https://keras.io/) Use to build model
* [Numpy](http://www.numpy.org/) Use to build model
* [PIL](https://pillow.readthedocs.io/en/3.1.x/index.html) Use to load ans save images
* [Scipy](https://www.scipy.org/about.html#) Use to minimize loss function

# Running

```
python3 neuralStyleTransfer.py
```
You need to change the names of images to load (base image and style image) in the code.

# Output example

![Screenshot](images/img.jpg)
Pic by [Mona Boitière](https://www.instagram.com/monaboitiere/)

![Screenshot](style1.jpg) ![Screenshot](style1.gif)
(Vincent van Gogh)
![Screenshot](style2.jpg) ![Screenshot](style2.gif)
(Don't know the artist)
![Screenshot](style4.jpg) ![Screenshot](style4.gif)
(David Hockney)

I then tried to transfer my style (from my drawings) to another picture :
![Screenshot](img2.jpg)
Pic from 'Drinking Buddies' movie

![Screenshot](couleur09.jpg) ![Screenshot](couleur09.gif)
![Screenshot](couleur35.jpg) ![Screenshot](couleur35.gif)
![Screenshot](couleur12.jpg) ![Screenshot](couleur12.gif)
![Screenshot](couleur41.jpg) ![Screenshot](couleur41.gif)

# Thanks

* [Keras Neural Transfer Style code] (https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
