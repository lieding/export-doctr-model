from .crnn import crnn_mobilenet_v3_large

def execute():
  model = crnn_mobilenet_v3_large(pretrained=True, exportable=True)
  return model