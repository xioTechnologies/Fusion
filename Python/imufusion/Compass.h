#ifndef COMPASS_H
#define COMPASS_H

#include "../../Fusion/Fusion.h"
#include "Convention.h"
#include "NpArray.h"
#include <Python.h>

static PyObject *compass(PyObject *self, PyObject *args) {
    PyObject *accelerometer_object;
    PyObject *magnetometer_object;
    int convention_int = fusionAhrsDefaultSettings.convention;

    if (PyArg_ParseTuple(args, "OO|i", &accelerometer_object, &magnetometer_object, &convention_int) == 0) {
        return NULL;
    }

    FusionVector accelerometer;

    if (np_array_1x3_to(accelerometer.array, accelerometer_object) != 0) {
        return NULL;
    }

    FusionVector magnetometer;

    if (np_array_1x3_to(magnetometer.array, magnetometer_object) != 0) {
        return NULL;
    }

    FusionConvention convention;

    if (convention_from(&convention, convention_int) != 0) {
        return NULL;
    }

    const float heading = FusionCompass(accelerometer, magnetometer, convention);

    return PyFloat_FromDouble((double) heading);
}

static PyMethodDef compass_methods[] = {
    {"compass", (PyCFunction) compass, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
