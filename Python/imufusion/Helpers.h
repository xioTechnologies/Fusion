#ifndef HELPERS_H
#define HELPERS_H

#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>

static char *const parse_array(float *const destination, const PyArrayObject *const array, const int size) {
    if (PyArray_NDIM(array) != 1) {
        return "Array dimensions is not 1";
    }

    if (PyArray_SIZE(array) != size) {
        static char string[32];
        snprintf(string, sizeof(string), "Array size is not %u", size);
        return string;
    }

    int offset = 0;

    for (int index = 0; index < size; index++) {
        PyObject *object = PyArray_GETITEM(array, PyArray_BYTES(array) + offset);

        destination[index] = (float) PyFloat_AsDouble(object);
        Py_DECREF(object);

        if (PyErr_Occurred()) {
            return "Invalid array element type";
        }

        offset += (int) PyArray_STRIDE(array, 0);
    }

    return NULL;
}

#endif
