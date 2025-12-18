#ifndef OFFSET_H
#define OFFSET_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

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
    PyObject *gyroscope_object;

    if (PyArg_ParseTuple(args, "O", &gyroscope_object) == 0) {
        return NULL;
    }

    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, gyroscope_object) != 0) {
        return NULL;
    }

    const FusionVector compensated_gyroscope = FusionOffsetUpdate(&self->offset, gyroscope);

    return np_array_1x3_from(compensated_gyroscope.array);
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
