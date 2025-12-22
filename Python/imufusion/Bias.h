#ifndef BIAS_H
#define BIAS_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionBias bias;
} Bias;

static PyObject *bias_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    unsigned int sample_rate;

    if (PyArg_ParseTuple(args, "I", &sample_rate) == 0) {
        return NULL;
    }

    Bias *const self = (Bias *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    FusionBiasInitialise(&self->bias, sample_rate);
    return (PyObject *) self;
}

static void bias_free(Bias *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *bias_update(Bias *self, PyObject *args) {
    PyObject *gyroscope_object;

    if (PyArg_ParseTuple(args, "O", &gyroscope_object) == 0) {
        return NULL;
    }

    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, gyroscope_object) != 0) {
        return NULL;
    }

    const FusionVector compensated_gyroscope = FusionBiasUpdate(&self->bias, gyroscope);

    return np_array_1x3_from(compensated_gyroscope.array);
}

static PyMethodDef bias_methods[] = {
    {"update", (PyCFunction) bias_update, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

static PyTypeObject bias_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.Bias",
    .tp_basicsize = sizeof(Bias),
    .tp_dealloc = (destructor) bias_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = bias_new,
    .tp_methods = bias_methods,
};

#endif
