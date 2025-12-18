#ifndef COMPASS_H
#define COMPASS_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

static PyObject *compass(PyObject *self, PyObject *args) {
    FusionConvention convention;
    PyArrayObject *accelerometer_array;
    PyArrayObject *magnetometer_array;

    if (PyArg_ParseTuple(args, "iO!O!", &convention, &PyArray_Type, &accelerometer_array, &PyArray_Type, &magnetometer_array) == 0) {
        return NULL;
    }

    FusionVector accelerometer_vector;
    FusionVector magnetometer_vector;

    const char *error = parse_array(accelerometer_vector.array, accelerometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    error = parse_array(magnetometer_vector.array, magnetometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    return PyFloat_FromDouble((double) FusionCompass(convention, accelerometer_vector, magnetometer_vector));
}

static PyMethodDef compass_methods[] = {
    {"compass", (PyCFunction) compass, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
