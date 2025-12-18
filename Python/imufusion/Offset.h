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

    if (PyArg_ParseTuple(args, "I", &sample_rate) == 0) {
        return NULL;
    }

    Offset *const self = (Offset *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    FusionOffsetInitialise(&self->offset, sample_rate);
    return (PyObject *) self;
}

static void offset_free(Offset *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *offset_update(Offset *self, PyObject *args) {
    PyArrayObject *input_array;

    if (PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array) == 0) {
        return NULL;
    }

    FusionVector input_vector;

    const char *error = parse_array(input_vector.array, input_array, 3);
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

static PyMethodDef offset_methods[] = {
    {"update", (PyCFunction) offset_update, METH_VARARGS, ""},
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
