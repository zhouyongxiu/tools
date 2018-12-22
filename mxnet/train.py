import argparse,logging,os
import mxnet as mx
import symbols.resnet as resnet
import mxnet.optimizer as optimizer
import numpy as np

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


def multi_factor_scheduler(begin_epoch, epoch_size, step=[20, 40, 60], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    # print (all_layers.list_outputs())
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='new_fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

def main():
    num_classes = 4
    gpus = None
    lr = 0.001
    step = [30, 60, 90, 120]
    gamma = 0.5
    mom = 0.9
    wd = 0.00001
    batch_size = 4
    epoch_size = 120
    bn_mom = 0.9
    data_channel = 3
    data_width = 224
    data_height = 224
    max_epoch = 200

    workspace = 512
    frequent = 20
    memonger = False
    # pre_train = ['resnet50/resnet-50', 0]
    pre_train = None
    fine_tuning = False
    arg_params = None
    aux_params = None

    train_data = "./data/train.lst"
    val_data = "./data/valid.lst"
    # train_idx = "../train.idx"
    # val_idx = "../valid.idx"
    model_prefix = "model/plate"

    if pre_train is None:
        symbol = resnet.get_symbol(num_classes, 50, '3,224,224')
        begin_epoch = 0
    else:
        symbol, arg_params, aux_params = mx.model.load_checkpoint(pre_train[0], pre_train[1])
        begin_epoch = pre_train[1]
    # mx.viz.plot_network(symbol).view()
    if fine_tuning:
        (symbol, arg_params) = get_fine_tune_model(symbol, arg_params, num_classes,  layer_name = 'flatten0')
    mx.viz.plot_network(symbol).view()
    # data_shape = {'data': (1, data_channel, data_height, data_width)}
    # arg_shape, out_shape, _ = symbol.infer_shape(**data_shape)
    # arg_name = symbol.list_arguments()
    # out_name = symbol.list_outputs()
    # log = zip(arg_name, arg_shape)
    # for message in log:
    #     print message
    # print({'input' : dict(zip(arg_name, arg_shape)), 'output': dict(zip(out_name, out_shape))})
    # mx.viz.plot_network(symbol).view()
    devs = mx.cpu() if gpus is None else [mx.gpu(i) for i in gpus]
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    if memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(batch_size, data_channel, data_height, data_width))
    train_iter = mx.image.ImageIter(
        path_imglist        = train_data,
        path_root           = './',
        # path_imgrec         = train_data,
        # path_imgidx         = train_idx,
        # data_name           = 'data',
        # label_name          = 'softmax_label',
        data_shape          = (data_channel, data_height, data_width),
        batch_size          = batch_size,
        resize              = 240,
        rand_crop           = True,
        rand_mirror         = True,
        shuffle             = True
        )
    val_iter = mx.image.ImageIter(
        path_imglist        = val_data,
        path_root           = './',
        # path_imgrec         = val_data,
        # path_imgidx         = val_idx,
        # data_name           = 'data',
        # label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = (data_channel, data_height, data_width),
        resize              = data_height,
        rand_crop           = False,
        rand_mirror         = False,
        shuffle             = True
        )

    model = mx.mod.Module(symbol=symbol,
                        context=devs,
                        data_names=['data'],
                        label_names=['softmax_label'])
    lr_scheduler = multi_factor_scheduler(0, epoch_size, step=step, factor=gamma)
    initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    optimizer_params = {
        'learning_rate': lr,
        'momentum': mom,
        'wd': wd,
        'lr_scheduler': lr_scheduler,
        # 'multi_precision': True
    }
    model.fit(train_iter,
              begin_epoch=begin_epoch,
              num_epoch=max_epoch,
              eval_data=val_iter,
              eval_metric='acc',
              optimizer='sgd',
              optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(batch_size, frequent),
              epoch_end_callback=checkpoint)
    metric = mx.metric.Accuracy()
    print (model.score(val_iter, metric))

if __name__ == "__main__":

    main()
