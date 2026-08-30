#ifndef CONVENTION_H
#define CONVENTION_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

static int convention_from(FusionConvention *const convention, const int convention_int) {
    switch (convention_int) {
        case FusionConventionNwu:
        case FusionConventionEnu:
        case FusionConventionNed:
            *convention = (FusionConvention) convention_int;
            return 0;
    }

    PyErr_SetString(PyExc_ValueError, "'convention' must be imufusion.CONVENTION_*");
    return -1;
}

static PyObject *convention_to_string(PyObject *null, PyObject *arg) {
    const long convention = PyLong_AsLong(arg);

    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyUnicode_FromString(FusionConventionToString((FusionConvention) convention));
}

static PyMethodDef convention_methods[] = {
    {"convention_to_string", (PyCFunction) convention_to_string, METH_O, ""},
    {NULL} /* sentinel */
};

#endif
