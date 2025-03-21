#ifndef OFFSET_H
#define OFFSET_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    FusionOffset offset;
} Offset;

static PyObject *offset_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    unsigned int sample_rate;

    const char *const error = PARSE_TUPLE(args, "I", &sample_rate);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    Offset *const self = (Offset *) subtype->tp_alloc(subtype, 0);
    FusionOffsetInitialise(&self->offset, sample_rate);
    return (PyObject *) self;
}

static void offset_free(Offset *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *offset_update(Offset *self, PyObject *args) {
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
    *output_vector = FusionOffsetUpdate(&self->offset, input_vector);

    const npy_intp dims[] = {3};
    PyObject *output_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, output_vector->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) output_array, NPY_ARRAY_OWNDATA);
    return output_array;
}


static PyObject *offset_update_batch(Offset *self, PyObject *args) {
    PyArrayObject *input_array;

    const char *error = PARSE_TUPLE(args, "O!", &PyArray_Type, &input_array);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    if (PyArray_TYPE(input_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "input_array must be np.float32");
        return NULL;
    }

    if (PyArray_NDIM(input_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 2-dimensional (n,3).");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(input_array);
    if (dims[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Input array's second dimension must be 3.");
        return NULL;
    }
    npy_intp n = dims[0];

    FusionVector input_vector, output_vector;

    const npy_intp out_dims[2] = { n, 3 };
    PyObject *output_array = PyArray_SimpleNew(2, out_dims, NPY_FLOAT);
    float *output_data = (float *) PyArray_DATA((PyArrayObject *)output_array);    

    float *input_data = (float *) PyArray_DATA(input_array);

    for (npy_intp i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            input_vector.array[j] = input_data[i * 3 + j];
        }

        // FusionVector *const output_vector = malloc(sizeof(FusionVector));
        // *output_vector = FusionOffsetUpdate(&self->offset, input_vector);
        output_vector = FusionOffsetUpdate(&self->offset, input_vector);

        for (int j = 0; j < 3; j++) {
            output_data[i * 3 + j] = output_vector.array[j];
        }
        // free(output_vector);
    }

    PyArray_ENABLEFLAGS((PyArrayObject *) output_array, NPY_ARRAY_OWNDATA);
    return output_array;
}

static PyMethodDef offset_methods[] = {
        {"update",       (PyCFunction) offset_update,       METH_VARARGS, ""},
        {"update_batch", (PyCFunction) offset_update_batch, METH_VARARGS, ""},
        {NULL} /* sentinel */
};

static PyTypeObject offset_object = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "imufusion.Offset",
        .tp_basicsize = sizeof(Offset),
        .tp_dealloc = (destructor) offset_free,
        .tp_new = offset_new,
        .tp_methods = offset_methods,
};

#endif
