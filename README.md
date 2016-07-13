# tf_toy_2d_classification

This is Tensorflow implementation of an simple and educational toy classifier (2 fully connected layers) using ConvNetJS: 
 http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

- Simply run `python train.py`
- `pts.npy` and `labels.npy` files are poins and labels respectively for training.
- If you want to generate new `pts.npy` and `labels.npy`, run `python toyinput.py`

![alt tag](https://github.com/ywpkwon/tf_toy_2d_classification/blob/master/fig1.eps)

For information, I refered a basic skeleton in http://stackoverflow.com/questions/34479872/why-is-tensorflow-100x-slower-than-convnetjs-in-this-simple-nn-example. I changed the loss function from `L2` to `cross entropy` (`train.py` has both), and added input generation and output visualization.
