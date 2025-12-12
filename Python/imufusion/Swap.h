#ifndef SWAP_H
#define SWAP_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

static PyObject *swap(PyObject *self, PyObject *args) {
    PyArrayObject *input_array;
    int alignment;

    const char *error = PARSE_TUPLE(args, "O!i", &PyArray_Type, &input_array, &alignment);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector input_vector;

    error = parse_array(input_vector.array, input_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector *const output_vector = malloc(sizeof(FusionVector));
    *output_vector = FusionSwap(input_vector, (FusionSwapAlignment) alignment);

    const npy_intp dims[] = {3};
    PyObject *output_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, output_vector->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) output_array, NPY_ARRAY_OWNDATA);
    return output_array;
}

static PyMethodDef swap_methods[] = {
        {"swap", (PyCFunction) swap, METH_VARARGS, ""},
        {NULL} /* sentinel */
};

#endif
