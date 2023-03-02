#ifndef COMPASS_H
#define COMPASS_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

static PyObject *compass_calculate_heading(PyObject *self, PyObject *args) {
    FusionConvention convention;
    PyArrayObject *accelerometer_array;
    PyArrayObject *magnetometer_array;

    const char *error = PARSE_TUPLE(args, "iO!O!", &convention, &PyArray_Type, &accelerometer_array, &PyArray_Type, &magnetometer_array);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector accelerometer_vector;
    FusionVector magnetometer_vector;

    error = parse_array(accelerometer_vector.array, accelerometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    error = parse_array(magnetometer_vector.array, magnetometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    return Py_BuildValue("f", FusionCompassCalculateHeading(convention, accelerometer_vector, magnetometer_vector));
}

static PyMethodDef compass_methods[] = {
        {"compass_calculate_heading", (PyCFunction) compass_calculate_heading, METH_VARARGS, ""},
        {NULL} /* sentinel */
};

#endif
