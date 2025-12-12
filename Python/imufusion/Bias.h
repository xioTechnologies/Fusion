#ifndef BIAS_H
#define BIAS_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    FusionBias bias;
} Bias;

static PyObject *bias_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    unsigned int sample_rate;

    const char *const error = PARSE_TUPLE(args, "I", &sample_rate);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    Bias *const self = (Bias *) subtype->tp_alloc(subtype, 0);
    FusionBiasInitialise(&self->bias, sample_rate);
    return (PyObject *) self;
}

static void bias_free(Bias *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *bias_update(Bias *self, PyObject *args) {
    PyArrayObject *input_array;

    const char *error = PARSE_TUPLE(args, "O!", &PyArray_Type, &input_array);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector input_vector;

    error = parse_array(input_vector.array, input_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector *const output_vector = malloc(sizeof(FusionVector));
    *output_vector = FusionBiasUpdate(&self->bias, input_vector);

    const npy_intp dims[] = {3};
    PyObject *output_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, output_vector->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) output_array, NPY_ARRAY_OWNDATA);
    return output_array;
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
        .tp_new = bias_new,
        .tp_methods = bias_methods,
};

#endif
