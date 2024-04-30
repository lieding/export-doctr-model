from .crnn import crnn_vgg16_bn
import tensorflow as tf

model = crnn_vgg16_bn(pretrained=True)

def execute():
  model = crnn_vgg16_bn(pretrained=True)
  tf.saved_model.save(model, "../output/crnn_vgg16_bn.h5")