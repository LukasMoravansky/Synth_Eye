## =========================================================================== ## 
# MIT License
# Copyright (c) 2025 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
## =========================================================================== ## 
# Author   : Roman Parak
# Email    : Roman.Parak@outlook.com
# Github   : https://github.com/rparak
# File Name: Model.py
## =========================================================================== ##

def Freeze_Backbone(trainer):
  """
  Description:
    Function to freeze the backbone layers of the model.

    Reference:
      https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/

    Note:
      The backbone of the model consists of layers 0-9 (10 layers).
  """

  # Number of layers to be frozen.
  num_frozen_layers = 10

  # Express the model.
  model = trainer.model

  print('[INFO] Freeze the backbone layers of the model:')
  freeze = [f'model.{x}.' for x in range(num_frozen_layers)]
  for k, v in model.named_parameters(): 
      v.requires_grad = True
      if any(x in k for x in freeze):
          print(f'[INFO] Freezing: {k}') 
          v.requires_grad = False