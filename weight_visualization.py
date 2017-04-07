import numpy as np
from keras import backend as k
from keras import applications
import matplotlib.pyplot as plt
k.set_image_dim_ordering('tf')

def deprocess_image(x):  # to standardize the given input
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    if k.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (k.sqrt(k.mean(k.square(x))) + 1e-5)

img_width = 150
img_height = 150
layer_name = 'block1_conv1'
n = 9
model = applications.VGG16(include_top=False, weights='imagenet')
model.summary()
weights = model.get_weights()
weights = weights[0]  # first convolutional layer
weights = deprocess_image(weights)  # for standardization
for i in range(1,n+1):
    plt.subplot(3, 3, i)
    plt.imshow(weights[:, :, :, i])
    plt.ylabel('%dth filter of %s layer' % (i, layer_name))
plt.suptitle('Filters of the %s layer' % layer_name)
plt.show()


input_img = model.input
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
layer_name = 'block1_conv1'

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (k.sqrt(k.mean(k.square(x))) + 1e-5)

kept_filters = []
for filter_index in range(0, n):

    print('Processing filter %d' % filter_index)

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if k.image_data_format() == 'channels_first':
        loss = k.mean(layer_output[:, filter_index, :, :])
    else:
        loss = k.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = k.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = k.function([input_img], [loss, grads])
    from matplotlib import pyplot as plt
    # step size for gradient ascent
    step = 1.
    from scipy import misc
    # we start from a gray image with some random noise
    if k.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        plt.subplot(3, 3, filter_index + 1)
        plt.imshow(img)
        plt.ylabel('%dth filter of %s  layer' % (filter_index, layer_name))
    else:
        plt.subplot(3, 3, filter_index + 1)
        plt.imshow(np.zeros((img_width,img_width)))
        plt.ylabel('Filter is unrelated')
plt.suptitle('Effects of Learned Filters on Randomly Generated Image')
plt.show()





