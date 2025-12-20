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

static int bias_set_settings(Bias *self, PyObject *value, void *closure) {
    if (PyObject_TypeCheck(value, &bias_settings_object) == 0) {
        PyErr_Format(PyExc_TypeError, "'settings' must be %s", bias_settings_object.tp_name);
        return -1;
    }

    FusionBiasSetSettings(&self->wrapped, &((BiasSettings *) value)->settings);
    return 0;
}

static PyObject *bias_update(Bias *self, PyObject *arg) {
    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, arg) != 0) {
        return NULL;
    }

    const FusionVector compensated_gyroscope = FusionBiasUpdate(&self->wrapped, gyroscope);

    return np_array_1x3_from(compensated_gyroscope.array);
}

static PyGetSetDef bias_get_set[] = {
    {"settings", NULL, (setter) bias_set_settings, "", NULL},
    {NULL} /* sentinel */
};

static PyMethodDef bias_methods[] = {
    {"update", (PyCFunction) bias_update, METH_O, ""},
    {NULL} /* sentinel */
};

static PyTypeObject bias_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.Bias",
    .tp_basicsize = sizeof(Bias),
    .tp_dealloc = (destructor) bias_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = bias_new,
    .tp_getset = bias_get_set,
    .tp_methods = bias_methods,
};

#endif
