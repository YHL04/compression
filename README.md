

# An attempt to achieve better compression through deep learning


## Idea

The idea is very simple, it uses a convolutional neural network to take in a 32x32x3
and output a 64x3 basis and 64x3 coefficients. The model is then trained using mean
squared error against the original image for lossy compression after reconstructing it
by multiplying the coefficients and the basis. The basis is a parameter inside the model
so it is the same across the entire dataset, so for compression all you need to store
is the coefficients which is 64x3 compared to the original input 32x32x3.
