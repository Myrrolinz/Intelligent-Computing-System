import hashlib
from collections import OrderedDict
import caffe.proto.caffe_pb2 as caffe_pb2
import numpy as np

def _save_weights(data, weightfile):
    """
    将权重数据数组data保存到指定的文件weightfile中
    """
    print('Save to ', weightfile)
    wsize = data.size
    weights = np.zeros((wsize + 4, ), dtype=np.int32)
    weights[0] = 0
    weights[1] = 1
    weights[2] = 0
    weights[3] = 0
    weights.tofile(weightfile)
    weights = np.fromfile(weightfile, dtype=np.float32)
    weights[4:] = data
    weights.tofile(weightfile)

def _save_cfg(blocks, cfgfile):
    """
    将神经网络配置块保存到文件
    """
    with open(cfgfile, 'w') as fp:
        for block in blocks:
            fp.write('[%s]\n' % (block['type']))
            for key, value in block.items():
                if key != 'type':
                    fp.write(f'{key}={value}\n')
            fp.write('\n')

def _print_cfg(blocks):
    """
    打印Darknet框架中的神经网络配置块
    """
    for block in blocks:
        print('[%s]' % (block['type']))
        for key, value in block.items():
            if key != 'type':
                print(f'{key}={value}')
        print('')
def _print_cfg_nicely(blocks):
    print('layer     filters    size              input                output')
    """
    prev_width：上一层网络的输出宽度
    prev_height：上一层网络的输出高度
    prev_filters：上一层网络的通道数
    """
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue

        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) / 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) / stride + 1
            height = (prev_height + 2 * pad - kernel_size) / stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d \
                x%4d   ->   %3d x %3d x%4d' %
                  (ind, 'conv', filters, kernel_size, kernel_size, stride,
                   prev_width, prev_height, prev_filters, width, height,
                   filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
            
        #输出为prev_width、prev_height 和 prev_filters 相乘   
        elif block['type'] == 'flatten': 
            flat_size = prev_width * prev_height * prev_filters
            print('%5d %-6s %10d' % (ind, 'flatten', flat_size))
            prev_width = flat_size
            prev_height = 1
            prev_filters = 1
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
            
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width / stride
            height = prev_height / stride
            print('%5d %-6s       %d x %d / %d   %3d \
                x %3d x%4d   ->   %3d x %3d x%4d' %
                  (ind, 'max', pool_size, pool_size, stride, prev_width,
                   prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->      %3d' %
                  (ind, 'avg', prev_width, prev_height, prev_filters,
                   prev_filters))
            prev_width = 1
            prev_height = 1
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->      %3d' %
                  (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->      %3d' %
                  (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width / stride
            height = prev_height / stride
            print('%5d %-6s             / %d   %3d x %3d \
                x%4d   ->   %3d x %3d x%4d' %
                  (ind, 'reorg', stride, prev_width, prev_height, prev_filters,
                   width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'region':
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s' % (ind, 'softmax'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->      %3d' %
                  (ind, 'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

def _parse_caffemodel(caffemodel):
    """
    解析caffe文件
    """
    model = caffe_pb2.NetParameter()
    print('Loading caffemodel: ', caffemodel)
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model

def _parse_prototxt(protofile):
    """
    解析 Caffe 模型的配置文件，并将其转化为一个有序字典的形式返回，
    其中包含模型的属性信息(props)和层信息(layers)
    """
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0:  # key: value
                # print line
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key in block:
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1:  # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block
    fp = open(protofile)
    props = OrderedDict()
    layers = []
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0:  # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if key in props:
                if type(props[key]) == list:
                    props[key].append(value)
                else:
                    props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1:  # blockname {
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props
    
def _generate_hash(weightfile, cfgfile):
    with open(weightfile, 'rb') as file:
        md5 = hashlib.md5()
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            md5.update(chunk)
    with open(cfgfile, 'rb') as file:
        md5 = hashlib.md5()
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            md5.update(chunk)
    hash_str = md5.hexdigest()
    return hash_str

def _check_hash(weightfile, cfgfile, hash_str):
    tmp_hash_str = _generate_hash(weightfile, cfgfile)
    print(tmp_hash_str == hash_str)
    assert tmp_hash_str == hash_str

class Caffe2Darknet:

    def get_attribute(self, n):
        if n in self.__dict__:
            return self.__dict__[n]
        else:
            return None

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.get_attribute('net') is not None:
            assert isinstance(self.get_attribute('net'), str)
            self.net = self.get_attribute('net')
        else:
            self.net = None
        if self.get_attribute('weight') is not None:
            assert isinstance(self.get_attribute('weight'), str)
            self.weight = self.get_attribute('weight')
        else:
            self.weight = None
        if self.get_attribute('out_file') is not None:
            assert isinstance(self.get_attribute('out_file'), str)
            self.out_file = self.get_attribute('out_file')
        else:
            self.out_file = None

    def convert(self):
        protofile = self.net
        caffemodel = self.weight
        model = _parse_caffemodel(caffemodel)
        layers = model.layer
        if len(layers) == 0:
            print('Using V1LayerParameter')
            layers = model.layers

        lmap = {}
        for l_name in layers:
            lmap[l_name.name] = l_name

        net_info = _parse_prototxt(protofile)
        props = net_info['props']

        wdata = []
        blocks = []
        block = OrderedDict()
        block['type'] = 'net'
        if 'input_shape' in props:
            block['batch'] = props['input_shape']['dim'][0]
            block['channels'] = props['input_shape']['dim'][1]
            block['height'] = props['input_shape']['dim'][2]
            block['width'] = props['input_shape']['dim'][3]
        else:
            block['batch'] = props['input_dim'][0]
            block['channels'] = props['input_dim'][1]
            block['height'] = props['input_dim'][2]
            block['width'] = props['input_dim'][3]
        if 'mean_file' in props:
            block['mean_file'] = props['mean_file']
        blocks.append(block)

        layers = net_info['layers']
        layer_num = len(layers)
        i = 0  # layer id
        layer_id = dict()
        layer_id[props['input']] = 0
        while i < layer_num:
            layer = layers[i]
            print(i, layer['name'], layer['type'])
            if layer['type'] == 'Convolution':
                if layer_id[layer['bottom']] != len(blocks) - 1:
                    block = OrderedDict()
                    block['type'] = 'route'
                    block['layers'] = \
                        str(layer_id[layer['bottom']] - len(blocks))
                    blocks.append(block)
                # assert(i+1 < layer_num and
                #   layers[i+1]['type'] == 'BatchNorm')
                # assert(i+2 < layer_num and
                #   layers[i+2]['type'] == 'Scale')
                conv_layer = layers[i]
                block = OrderedDict()
                block['type'] = 'convolutional'
                block['filters'] = conv_layer['convolution_param'][
                    'num_output']
                block['size'] = conv_layer['convolution_param']['kernel_size']
                block['stride'] = conv_layer['convolution_param']['stride']
                block['pad'] = '1'
                last_layer = conv_layer
                m_conv_layer = lmap[conv_layer['name']]
                if i + 2 < layer_num and layers[
                        i + 1]['type'] == 'BatchNorm' and layers[
                            i + 2]['type'] == 'Scale':
                    print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
                    print(i + 2, layers[i + 2]['name'], layers[i + 2]['type'])
                    block['batch_normalize'] = '1'
                    bn_layer = layers[i + 1]
                    scale_layer = layers[i + 2]
                    last_layer = scale_layer
                    m_scale_layer = lmap[scale_layer['name']]
                    m_bn_layer = lmap[bn_layer['name']]
                    wdata += list(m_scale_layer.blobs[1].data)
                    wdata += list(m_scale_layer.blobs[0].data)
                    wdata += (np.array(m_bn_layer.blobs[0].data) /
                              m_bn_layer.blobs[2].data[0]).tolist()
                    wdata += (np.array(m_bn_layer.blobs[1].data) /
                              m_bn_layer.blobs[2].data[0]).tolist()
                    i = i + 2
                else:
                    wdata += list(m_conv_layer.blobs[1].data)
                wdata += list(m_conv_layer.blobs[0].data)

                if i + 1 < layer_num and layers[i + 1]['type'] == 'ReLU':
                    print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
                    act_layer = layers[i + 1]
                    block['activation'] = 'relu'
                    top = act_layer['top']
                    layer_id[top] = len(blocks)
                    blocks.append(block)
                    i = i + 1
                else:
                    block['activation'] = 'linear'
                    top = last_layer['top']
                    layer_id[top] = len(blocks)
                    blocks.append(block)
                i = i + 1
            elif layer['type'] == 'Pooling':
                assert (layer_id[layer['bottom']] == len(blocks) - 1)
                block = OrderedDict()
                if layer['pooling_param']['pool'] == 'AVE':
                    block['type'] = 'avgpool'
                elif layer['pooling_param']['pool'] == 'MAX':
                    block['type'] = 'maxpool'
                    block['size'] = layer['pooling_param']['kernel_size']
                    block['stride'] = layer['pooling_param']['stride']
                    if 'pad' in layer['pooling_param']:
                        pad = int(layer['pooling_param']['pad'])
                        if pad > 0:
                            block['pad'] = '1'
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
                
            elif layer['type'] == 'Flatten':
                assert (layer_id[layer['bottom']] == len(blocks) - 1)
                block = OrderedDict()
                #block['type'] = 'flatten'
                block['type'] = 'connected'
                block['output'] = 512
                block['activation'] = 'linear'
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            elif layer['type'] == 'Eltwise':
                bottoms = layer['bottom']
                bottom1 = layer_id[bottoms[0]] - len(blocks)
                bottom2 = layer_id[bottoms[1]] - len(blocks)
                assert (bottom1 == -1 or bottom2 == -1)
                from_id = bottom2 if bottom1 == -1 else bottom1
                block = OrderedDict()
                block['type'] = 'shortcut'
                block['from'] = str(from_id)
                assert (i + 1 < layer_num and layers[i + 1]['type'] == 'ReLU')
                block['activation'] = 'relu'
                top = layers[i + 1]['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 2
            elif layer['type'] == 'InnerProduct':
                assert (layer_id[layer['bottom']] == len(blocks) - 1)
                block = OrderedDict()
                block['type'] = 'connected'
                block['output'] = layer['inner_product_param']['num_output']
                m_fc_layer = lmap[layer['name']]
                wdata += list(m_fc_layer.blobs[1].data)
                wdata += list(m_fc_layer.blobs[0].data)
                if i + 1 < layer_num and layers[i + 1]['type'] == 'ReLU':
                    act_layer = layers[i + 1]
                    block['activation'] = 'relu'
                    top = act_layer['top']
                    layer_id[top] = len(blocks)
                    blocks.append(block)
                    i = i + 2
                else:
                    block['activation'] = 'linear'
                    top = layer['top']
                    layer_id[top] = len(blocks)
                    blocks.append(block)
                    i = i + 1
            elif layer['type'] == 'Softmax':
                assert (layer_id[layer['bottom']] == len(blocks) - 1)
                block = OrderedDict()
                block['type'] = 'softmax'
                block['groups'] = 1
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            else:
                print('unknown type %s' % layer['type'])
                if layer_id[layer['bottom']] != len(blocks) - 1:
                    block = OrderedDict()
                    block['type'] = 'route'
                    block['layers'] = str(layer_id[layer['bottom']] -
                                          len(blocks))
                    blocks.append(block)
                block = OrderedDict()
                block['type'] = layer['type']
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)

                i = i + 1

        print('done')
        assert self.out_file is not None
        weightfile = self.out_file + '.weights'
        cfgfile = self.out_file + '.cfg'
        _save_weights(np.array(wdata), weightfile)
        _save_cfg(blocks, cfgfile)
        _print_cfg(blocks)
        _print_cfg_nicely(blocks)
        print('Hash of Darknet model has been published: ',
              format(_generate_hash(weightfile, cfgfile)))
