import sys
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

filename = sys.argv[1]
outputfile = sys.argv[2]

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.savefig(outputfile)
	pyplot.close()

model = load_model(filename)
latent_points = generate_latent_points(100, 25)
X = model.predict(latent_points)
save_plot(X, 5)
