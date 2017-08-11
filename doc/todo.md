# TODO

## impl

+ input shape inference w/ `leaf.Sequetial`
+ automatic nice naming shared vars
+ pretty print Leaf objects
+ Conv2D, Conv1D weight transposed (wrong for Xavir inits)
+ CUDNN LSTM (`return xs, h_next, c_next`)
+ General RNN leaf
``` python
RNN(mode="LSTM", layers=4, n_input=5, n_hidden=3,
    dropout=2, bidirectional=True, projections=True
    skip=[0, 0, 2, 2], impl=RNNImpl.cudnn)
```
+ masking op
+ make optimizer's lr shared
+ yellowfin optimizer https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/tuner_utils/yellowfin.py
+ pytorch's reinforce & gym example
+ seq2seq and NMT example
+ conv seq2seq


## test

+ CUDNN GRU stacked forward/backward
+ CUDNN LSTM
+ ConvLSTM, ConvGRU
