#ifndef BIAS_H
#define BIAS_H

#include "../../Fusion/Fusion.h"
#include "BiasSettings.h"
#include "NpArray.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionBias wrapped;
} Bias;

static PyObject *bias_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    if (PyArg_ParseTuple(args, "") == 0) {
        return NULL;
    }

    Bias *const self = (Bias *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    FusionBiasInitialise(&self->wrapped);
    return (PyObject *) self;
}

static void bias_free(Bias *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *bias_set_settings(Bias *self, PyObject *arg) {
    BiasSettings *settings;

    if (PyArg_Parse(arg, "O!", &bias_settings_object, &settings) == 0) {
        return NULL;
    }

    FusionBiasSetSettings(&self->wrapped, &settings->wrapped);
    Py_RETURN_NONE;
}

static PyObject *bias_update(Bias *self, PyObject *arg) {
    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, arg) != 0) {
        return NULL;
    }

    const FusionVector corrected_gyroscope = FusionBiasUpdate(&self->wrapped, gyroscope);

    return np_array_1x3_from(corrected_gyroscope.array);
}

static PyObject *bias_get_offset(Bias *self, PyObject *args) {
    const FusionVector offset = FusionBiasGetOffset(&self->wrapped);

    return np_array_1x3_from(offset.array);
}

static PyObject *bias_set_offset(Bias *self, PyObject *arg) {
    FusionVector offset;

    if (np_array_1x3_to(offset.array, arg) != 0) {
        return NULL;
    }

    FusionBiasSetOffset(&self->wrapped, offset);
    Py_RETURN_NONE;
}

static PyMethodDef bias_methods[] = {
    {"set_settings", (PyCFunction) bias_set_settings, METH_O, ""},
    {"update", (PyCFunction) bias_update, METH_O, ""},
    {"get_offset", (PyCFunction) bias_get_offset, METH_NOARGS, ""},
    {"set_offset", (PyCFunction) bias_set_offset, METH_O, ""},
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
