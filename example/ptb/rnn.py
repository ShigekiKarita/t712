import theano
import theano.tensor as tt
import t721.leaf as L
import t721.initializer as I


class RNNModel(L.Leaf):
    def __init__(self, rnn_type, ntokens, emsize, nhid, nlayers, dropout, tied, optimizer):
        self.rnn_type = rnn_type
        self.nlayers = 1 # nlayers
        self.nhid = nhid
        self.optimizer = optimizer

        xs = tt.lmatrix("xs")
        self.embed = L.Embed(ntokens, emsize)
        self.dropout = L.Dropout(dropout)
        h_init = tt.tensor3("h_init")
        impl = L.RNNImpl.fused
        if rnn_type == "GRU" and nlayers == 2:
            self.gru1 = L.GRU(emsize, nhid, impl=impl)
            self.gru2 = L.GRU(nhid, nhid, impl=impl)
        else:
            raise NotImplementedError()
        self.fc = L.Linear(nhid, ntokens)
        if tied:
            self.fc.weight = self.embed.weight.T

        es = self.embed(xs.reshape([-1])).reshape([*xs.shape, emsize])
        ds = self.dropout(es)
        hs1, h1_next = self.gru1(ds, h_init[0])
        ds = self.dropout(hs1)
        hs2, h2_next = self.gru2(ds, h_init[1])
        ds = self.dropout(hs2)
        ys = self.fc(ds.reshape([-1, nhid]))
        ys = tt.nnet.softmax(ys)
        ts = tt.lvector("ts")
        loss = tt.mean(tt.nnet.categorical_crossentropy(ys, ts))
        h_next = tt.stack([h1_next, h2_next])

        print("compiling test function")
        self.test = theano.function([xs, ts, h_init], [loss, h_next],
                                    givens={self.is_train: tt.numpy.int8(False)})
        print("compiling train function")
        self.train = theano.function([xs, ts, h_init], [loss, h_next],
                                     updates=self.optimize(loss, optimizer),
                                     givens={self.is_train: tt.numpy.int8(True)})

    def init_hidden(self, bsz):
        zeros = I.Constant(0.0)
        if self.rnn_type == 'LSTM':
            return (zeros((self.nlayers, bsz, self.nhid)),
                    zeros((self.nlayers, bsz, self.nhid)))
        else:
            return zeros((self.nlayers, bsz, self.nhid))
