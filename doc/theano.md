# Theano tips


## 注意点

+ ややこしいけど、sandbox.cudaが旧バックエンドでgpuarrayが新バックエンド
+ GPUバックエンドが変わったので設定が違う 
    + -v0.8 `device=gpu` -> v0.9- `device=cuda`
    + -v0.8 `lib.cnmem=1.0` -> v0.9- `gpuarray.preallocate=1.0`
+ 新バックエンドではcuDNNのRNNもサポート
    + http://deeplearning.net/software/theano/library/gpuarray/dnn.html#theano.gpuarray.dnn.RNNBlock
    + 使い方がわからん (これをラップしたOpはまだない v0.9)

## CUDNN RNN support

### ドキュメント

(RNNと一体化されたDropoutを除く) 殆どの機能がサポートされている。
まともなドキュメントはないが、以下で断片的な情報が得られる

+ https://github.com/jiangnanhugo/bso/blob/master/layers/rnnblock.py
    + 実際にtっかっている人

+ http://deeplearning.net/software/theano/library/gpuarray/dnn.html#theano.gpuarray.dnn.RNNBlock
    + まともなドキュメントを書くべきとある (1年前から放置)
    + input_mode に関しては要調査
        ```
        rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
            See cudnn documentation for ``cudnnRNNMode_t``.
    
        input_mode : {'linear', 'skip'}
            linear: input will be multiplied by a biased matrix
            skip: No operation is performed on the input.
            The size must match the hidden size.
        ```
+ theano.gpuarray.tests.rnn_support
    + t721のベースにしたミニマルなラッパークラスがある
    + LSTMやGRUのシンボル定義がある
    + たぶん、このコードは scan の updates を other_updates に登録し忘れている
+ theano.gpuarray.tests.test_dnn
    + test_dnn_rnn_gru
        + このテストケースだけで、RNNBlockの大体の使い方がわかる
            ```python
            rnnb = dnn.RNNBlock(theano.config.floatX, hidden_dim, depth, 'gru')
            psize = rnnb.get_param_size([batch_size, input_dim])
            params_cudnn = gpuarray_shared_constructor(
                np.zeros((psize,), dtype=theano.config.floatX))
            # params_cudnn の配置はシンボル定義と同じ
            dnn_params = rnnb.split_params(params_cudnn, i,
                                          [batch_size, input_dim])
            for j, p in enumerate(dnn_params):
                p[:] = symbol_gru.params[j].get_value(borrow=True,
                                                      return_internal_type=True)
            y, hy = rnnb.apply(params_cudnn, X, h0)
            ```
    + test_dnn_rnn_gru_bidi
        + 出力 y の形は (timesteps, batch_size, 2 * hidden_dim)
        + 状態 h の形は (2 * depth, batch_size, hidden_dim)
        + unidirectional なときも同様
    + test_dnn_rnn_lstm
        + `y, hy, cy = rnnb.apply(params_cudnn, X, h0, c0)`


### TODO

+ CPU互換な演算(シンボル定義)との相互変換ができると嬉しい
    + シンボル定義クラスの中、`theano.config.mode`と`theano.gpuarray.dnn.dnn_available`でどちらを使うか判別?
    + `theano.shared`の共有は文字通り、普通にやれば簡単だと思う (bidirectionalは要テスト)
+ シンボル定義で scan の updates を捨てているが、もしかすると空？ (テストは通っている)
+ この手作業でparam設定する方法を両立したい (self.paramがなければshared変数を探す?)
+ other_updatesはこれで良い。全部dict


## THEANO_FLAG, .theanorc の設定

### 設定方法

1. `theano.config.<property> = ... pythonグローバル変数` 
2. `THEANO_FLAGS=... 環境変数` 
3. `.theanorc ファイル` 

上記の順番で優先される(theano.configはread-onlyな変数もある)


### 最速の設定

たぶん

```ini
[global]
# theano.tensor.matrix(dtype) とかで使われる値
floatX = float32
# cpu とか opencl が使える。 gpu (昔のCUDAバックエンド) は非推奨
device = cuda
# あらゆる最適化を実行する、エラーメッセージは最悪
mode = FAST_RUN

[nvcc]
# 精度が犠牲になる。nanがでたらFalseにすべき
fastmath = True

[dnn]
# 最初の一回をベンチマークして選択 (guess_onceだとヒューリスティック)
conv.algo_fwd = time_once

[gcc]
cxxflags = -O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer

[gpuarray]
# GPUのMallocは遅いので最初にプールしておく量(0.0-1.0の割合かMBで指定)
preallocate = 1.0
```

convolution のアルゴリズム詳細 http://deeplearning.net/software/theano/library/gpuarray/dnn.html
GCC の最速設定 http://deeplearning.net/software/theano/faq.html#faster-gcc-optimization

### 開発用の設定

DebugModeはCPUだけで動かすのが無難

```ini
[global]
# theano.tensor.matrix(dtype) とかで使われる値
floatX = float32
# cpu とか opencl が使える。 gpu (昔のCUDAバックエンド) は非推奨
device = cpu
# あらゆる最適化をオフ、エラーもpythonのスタックトレースが効く
mode = DebugMode
# とにかく早く色々試したいならこっち
# mode = FAST_COMPILE
```


#### マルチGPU

http://deeplearning.net/software/theano/tutorial/using_multi_gpu.html

まずは設定ファイル(実デバイス→theanoのデバイス)

```ini
[global]
contexts=dev0->cuda0;dev1->cuda1

[dnn]
conv.algo_fwd = deterministic
```


#### マルチノード (MPI)

モデル並列は簡単そう

## プロファイリング

+ 全体のプロファイル `THEANO_FLAGS="profile=True,profile_memory=True,profile_optimizer=True"`
+ 関数単位のプロファイル `f = theano.function(..., profile=True); f.profile.summary()`

http://deeplearning.net/software/theano/tutorial/profiling.html#tut-profiling

## theano.function

## theano.scan


## その他

+ C code op http://deeplearning.net/software/theano/extending/fibby.html
+ CUDA kernel op http://deeplearning.net/software/theano/extending/extending_theano_gpu.html
+ serialization http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
+ theano.signal 便利そう
    http://deeplearning.net/software/theano/library/tensor/signal/index.html
+ theano.opt 手作業で最適化？
    http://deeplearning.net/software/theano/library/tensor/opt.html
+ theano.tensor.fft
    http://deeplearning.net/software/theano/library/tensor/fft.html
+ theano.gradの応用 (Hessian や Hessian x vなど)
    http://deeplearning.net/software/theano/tutorial/gradients.html
