import keras
from keras import backend
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import time
from scipy.optimize import fmin_l_bfgs_b

#######################     Model     ##########################################

# Open base image and preprocess
image_path = 'img.jpg'
image = Image.open(image_path).convert('RGB')
image = image.resize((512, 512))
img_input = np.array(image)
img_input = img_input.astype('float32')
img_input = np.expand_dims(img_input, axis=0)
img_input = preprocess_input(img_input)

# Open style image and preprocess
image_path = 'style.jpg'
image = Image.open(image_path).convert('RGB')
image = image.resize((512, 512))
img_style = np.array(image)
img_style = img_style.astype('float32')
img_style = np.expand_dims(img_style, axis=0)
img_style = preprocess_input(img_style)

# deprocess image function
def deprocess_image(x):
    x = x.reshape((512, 512, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# define variable for tensorflow graph
img_input = backend.variable(img_input)
img_style = backend.variable(img_style)
combination_image = backend.placeholder((1, 512, 512, 3))

# concatenate en 1 tensor
input_tensor = backend.concatenate([img_input,
                                    img_style,
                                    combination_image], axis=0)

# pretrained model
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('Model loaded.')

# Get model's layers
layers = dict([(layer.name, layer.output) for layer in model.layers])
print(layers)

###########################    loss   ##########################################
content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1.0

loss = backend.variable(0.)

#### loss feature
# based on intuition that images with similar content will have
# similar representation in the higher layers
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layer_features = layers['block5_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_image_features,
                                      combination_features)
#### loss style
# the gram matrix of an image tensor (capture style)
def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram
# style loss uses lower layers which capture low-level features
# based on the gram matrices of feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    assert backend.ndim(style) == 3
    assert backend.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 512 * 512
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

height=512
width=512

#### total loss
# minimize loss by changing input
def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))
loss += total_variation_weight * total_variation_loss(combination_image)

# define gradient
grads = backend.gradients(loss, combination_image)
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
f_outputs = backend.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


##########################   Training   ########################################
image_path = 'img.jpg'
image = Image.open(image_path).convert('RGB')
image = image.resize((512, 512))
img_input = np.array(image)
img_input = img_input.astype('float32')
img_input = np.expand_dims(img_input, axis=0)
x = preprocess_input(img_input)

iterations = 10

# fmin_l_bfgs_b minimize evaluator.loss func using L-BFGS-B,
# x.flatten -> initial guess, fprime -> the gradient of function
# maxfun -> max number of func evaluation
# returns estimated position of minimum, value of func at minimum
# and information dictionary
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    # save generated image
    img = deprocess_image(x.copy())
    img = Image.fromarray(img)
    imname = "image"+str(i)+".png"
    img.save(imname, "PNG")
    print(img)
