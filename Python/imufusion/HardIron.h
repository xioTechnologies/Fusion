#ifndef HARD_IRON_H
#define HARD_IRON_H

#include "../../Fusion/Fusion.h"
#include "HardIronSettings.h"
#include "NpArray.h"
#include "Progress.h"
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

static PyObject *hard_iron_set_settings(HardIron *self, PyObject *arg) {
    if (PyObject_TypeCheck(arg, &hard_iron_settings_object) == 0) {
        PyErr_Format(PyExc_TypeError, "'settings' must be %s", hard_iron_settings_object.tp_name);
        return NULL;
    }

    FusionHardIronSetSettings(&self->hardIron, &((HardIronSettings *) arg)->settings);
    Py_RETURN_NONE;
}

static PyObject *hard_iron_update(HardIron *self, PyObject *arg) {
    FusionVector magnetometer;

    if (np_array_1x3_to(magnetometer.array, arg) != 0) {
        return NULL;
    }

    const FusionResult result = FusionHardIronUpdate(&self->hardIron, magnetometer);

    if (result != FusionResultOk) {
        PyErr_SetString(PyExc_RuntimeError, FusionResultToString(result));
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *hard_iron_get_corrected_magnetometer(HardIron *self, PyObject *args) {
    const FusionVector correctedMagnetometer = FusionHardIronGetCorrectedMagnetometer(&self->hardIron);

    return np_array_1x3_from(correctedMagnetometer.array);
}

static PyObject *hard_iron_get_offset(HardIron *self, PyObject *args) {
    const FusionVector offset = FusionHardIronGetOffset(&self->hardIron);

    return np_array_1x3_from(offset.array);
}

static PyObject *hard_iron_set_offset(HardIron *self, PyObject *arg) {
    FusionVector offset;

    if (np_array_1x3_to(offset.array, arg) != 0) {
        return NULL;
    }

    FusionHardIronSetOffset(&self->hardIron, offset);
    Py_RETURN_NONE;
}

static PyObject *hard_iron_start(HardIron *self, PyObject *args) {
    FusionHardIronStart(&self->hardIron);
    Py_RETURN_NONE;
}

static PyObject *hard_iron_get_progress(HardIron *self, PyObject *args) {
    const FusionProgress progress = FusionHardIronGetProgress(&self->hardIron);

    return progress_from(&progress);
}

static PyObject *hard_iron_complete(HardIron *self, PyObject *args) {
    const FusionResult result = FusionHardIronComplete(&self->hardIron);

    if (result != FusionResultOk) {
        PyErr_SetString(PyExc_RuntimeError, FusionResultToString(result));
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *hard_iron_abort(HardIron *self, PyObject *args) {
    const FusionResult result = FusionHardIronAbort(&self->hardIron);

    if (result != FusionResultOk) {
        PyErr_SetString(PyExc_RuntimeError, FusionResultToString(result));
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *hard_iron_get_samples(HardIron *self, PyObject *args) {
    const FusionVector *samples;
    int numberOfSamples;

    const FusionResult result = FusionHardIronGetSamples(&self->hardIron, &samples, &numberOfSamples);

    if (result != FusionResultOk) {
        PyErr_SetString(PyExc_RuntimeError, FusionResultToString(result));
        return NULL;
    }

    return np_array_nx3_from(samples->array, numberOfSamples);
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

static PyMethodDef hard_iron_methods[] = {
    {"set_settings", (PyCFunction) hard_iron_set_settings, METH_O, ""},
    {"update", (PyCFunction) hard_iron_update, METH_O, ""},
    {"get_corrected_magnetometer", (PyCFunction) hard_iron_get_corrected_magnetometer, METH_NOARGS, ""},
    {"get_offset", (PyCFunction) hard_iron_get_offset, METH_NOARGS, ""},
    {"set_offset", (PyCFunction) hard_iron_set_offset, METH_O, ""},
    {"start", (PyCFunction) hard_iron_start, METH_NOARGS, ""},
    {"get_progress", (PyCFunction) hard_iron_get_progress, METH_NOARGS, ""},
    {"complete", (PyCFunction) hard_iron_complete, METH_NOARGS, ""},
    {"abort", (PyCFunction) hard_iron_abort, METH_NOARGS, ""},
    {"get_samples", (PyCFunction) hard_iron_get_samples, METH_NOARGS, ""},
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
    .tp_methods = hard_iron_methods,
};

#endif
