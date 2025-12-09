#ifndef HARD_IRON_H
#define HARD_IRON_H

#include "../../Fusion/Fusion.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

static PyObject *hard_iron_solve(PyObject *self, PyObject *args) {
    FusionVector samples[] = {
        {-0.377f, 0.277f, 0.677f},
        {0.777f, -0.877f, 0.677f},
        {0.777f, 0.277f, -0.477f},
        {-0.377f, -0.877f, 0.677f},
        {-0.377f, 0.277f, -0.477f},
        {0.777f, -0.877f, -0.477f},
        {-0.377f, -0.877f, -0.477f},
    }; // TODO: receive samples as function argument

    const int numberOfSamples = sizeof(samples) / sizeof(samples[0]);

    FusionVector *const offset = malloc(sizeof(FusionVector)); // TODO: fix memory leak

    const FusionResult result = FusionHardIronSolve(samples, numberOfSamples, offset);

    if (result != FusionResultOk) {
        PyErr_SetString(PyExc_RuntimeError, FusionResultToString(result));
        return NULL;
    }

    const npy_intp dims[] = {3};
    PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, offset->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyMethodDef hard_iron_methods[] = {
    {"hard_iron_solve", (PyCFunction) hard_iron_solve, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
