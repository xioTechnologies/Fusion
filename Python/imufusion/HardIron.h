#ifndef HARD_IRON_H
#define HARD_IRON_H

#include "../../Fusion/Fusion.h"
#include "HardIronSettings.h"
#include "NpArray.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionHardIron hardIron;
} HardIron;

static PyObject *hard_iron_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    if (PyArg_ParseTuple(args, "") == 0) {
        return NULL;
    }

    HardIron *const self = (HardIron *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    FusionHardIronInitialise(&self->hardIron);
    return (PyObject *) self;
}

static void hard_iron_free(HardIron *self) {
    Py_TYPE(self)->tp_free(self);
}

static int hard_iron_set_settings(HardIron *self, PyObject *value, void *closure) {
    if (PyObject_TypeCheck(value, &hard_iron_settings_object) == 0) {
        PyErr_Format(PyExc_TypeError, "'settings' must be %s", ahrs_settings_object.tp_name);
        return -1;
    }

    FusionHardIronSetSettings(&self->hardIron, &((HardIronSettings *) value)->settings);
    return 0;
}

static PyObject *hard_iron_update(HardIron *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *magnetometer_object;

    if (PyArg_ParseTuple(args, "OO", &gyroscope_object, &magnetometer_object) == 0) {
        return NULL;
    }

    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, gyroscope_object) != 0) {
        return NULL;
    }

    FusionVector magnetometer;

    if (np_array_1x3_to(magnetometer.array, magnetometer_object) != 0) {
        return NULL;
    }

    FusionHardIronUpdate(&self->hardIron, gyroscope, magnetometer);

    const FusionVector compensated_magnetometer = FusionHardIronUpdate(&self->hardIron, gyroscope, magnetometer);

    return np_array_1x3_from(compensated_magnetometer.array);
}

static PyObject *hard_iron_solve(HardIron *null, PyObject *arg) {
    float *samples_array;
    int numberOfSamples;

    if (np_array_nx3_to(&samples_array, &numberOfSamples, arg) != 0) {
        return NULL;
    }

    const FusionVector *const samples = (FusionVector *) samples_array;

    FusionVector offset;

    const FusionResult result = FusionHardIronSolve(samples, numberOfSamples, &offset);

    PyMem_Free(samples_array);

    if (result != FusionResultOk) {
        PyErr_SetString(PyExc_RuntimeError, FusionResultToString(result));
        return NULL;
    }

    return np_array_1x3_from(offset.array);
}

static PyGetSetDef hard_iron_get_set[] = {
    {"settings", NULL, (setter) hard_iron_set_settings, "", NULL},
    {NULL} /* sentinel */
};

static PyMethodDef hard_iron_methods[] = {
    {"update", (PyCFunction) hard_iron_update, METH_VARARGS, ""},
    {"solve", (PyCFunction) hard_iron_solve, METH_O | METH_STATIC, ""},
    {NULL} /* sentinel */
};

static PyTypeObject hard_iron_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.HardIron",
    .tp_basicsize = sizeof(HardIron),
    .tp_dealloc = (destructor) hard_iron_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = hard_iron_new,
    .tp_getset = hard_iron_get_set,
    .tp_methods = hard_iron_methods,
};

#endif
