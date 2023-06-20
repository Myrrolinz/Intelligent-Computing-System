#include "flatten_layer.h"
#include "utils.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_flatten_layer(int batch, int inputs)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = FLATTEN;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)xcalloc(batch * inputs, sizeof(float));
    l.delta = (float*)xcalloc(batch * inputs, sizeof(float));

    l.forward = forward_flatten_layer;
    l.backward = backward_flatten_layer;
#ifdef GPU
    l.forward_gpu = forward_flatten_layer_gpu;
    l.backward_gpu = backward_flatten_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    fprintf(stderr, "Flatten Layer: %d inputs\n", inputs);
    return l;
}

void forward_flatten_layer(layer l, network state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
}

void backward_flatten_layer(layer l, network state)
{
    copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_flatten_layer_gpu(layer l, network state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
}

void backward_flatten_layer_gpu(layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
