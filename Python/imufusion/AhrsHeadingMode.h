#ifndef AHRS_HEADING_MODE_H
#define AHRS_HEADING_MODE_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

static int ahrs_heading_mode_from(FusionAhrsHeadingMode *const heading_mode, const int heading_mode_int) {
    switch (heading_mode_int) {
        case FusionAhrsHeadingModeMagnetic:
        case FusionAhrsHeadingModeRelative:
        case FusionAhrsHeadingModeExternal:
            *heading_mode = (FusionAhrsHeadingMode) heading_mode_int;
            return 0;
    }

    PyErr_SetString(PyExc_ValueError, "'heading_mode' must be imufusion.HEADING_MODE_*");
    return -1;
}

static PyObject *heading_mode_to_string(PyObject *null, PyObject *arg) {
    const long heading_mode = PyLong_AsLong(arg);

    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyUnicode_FromString(FusionAhrsHeadingModeToString((FusionAhrsHeadingMode) heading_mode));
}

static PyMethodDef ahrs_heading_mode_methods[] = {
    {"heading_mode_to_string", (PyCFunction) heading_mode_to_string, METH_O, ""},
    {NULL} /* sentinel */
};

#endif
