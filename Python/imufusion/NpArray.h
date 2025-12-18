#ifndef NP_ARRAY_H
#define NP_ARRAY_H

#include <numpy/arrayobject.h>
#include <Python.h>

static int np_array_1x3_to(float *const c_array, PyObject *object) {
    PyArrayObject *np_array = (PyArrayObject *) PyArray_FROM_OTF(object, NPY_FLOAT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

    if (np_array == NULL) {
        return 1;
    }

    if (PyArray_NDIM(np_array) != 1 || PyArray_DIM(np_array, 0) != 3) {
        PyErr_SetString(PyExc_TypeError, "Array must have shape (3,)");
        Py_DECREF(np_array);
        return 1;
    }

    memcpy(c_array, PyArray_DATA(np_array), sizeof(float) * 3);
    Py_DECREF(np_array);
    return 0;
}

static int np_array_1x4_to(float *const c_array, PyObject *object) {
    PyArrayObject *np_array = (PyArrayObject *) PyArray_FROM_OTF(object, NPY_FLOAT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

    if (np_array == NULL) {
        return 1;
    }

    if (PyArray_NDIM(np_array) != 1 || PyArray_DIM(np_array, 0) != 4) {
        PyErr_SetString(PyExc_TypeError, "Array must have shape (4,)");
        Py_DECREF(np_array);
        return 1;
    }

    memcpy(c_array, PyArray_DATA(np_array), sizeof(float) * 4);
    Py_DECREF(np_array);
    return 0;
}

static PyObject *np_array_1x3_from(const float *const c_array) {
    const npy_intp dimensions[] = {3};
    PyObject *np_array = PyArray_SimpleNew(1, dimensions, NPY_FLOAT);

    if (np_array == NULL) {
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) np_array), c_array, 3 * sizeof(float));
    return np_array;
}

static PyObject *np_array_1x4_from(const float *const c_array) {
    const npy_intp dimensions[] = {4};
    PyObject *np_array = PyArray_SimpleNew(1, dimensions, NPY_FLOAT);

    if (np_array == NULL) {
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) np_array), c_array, 4 * sizeof(float));
    return np_array;
}

static PyObject *np_array_3x3_from(const float *const c_array) {
    const npy_intp dimensions[] = {3, 3};
    PyObject *array = PyArray_SimpleNew(2, dimensions, NPY_FLOAT);

    if (array == NULL) {
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) array), c_array, 9 * sizeof(float));
    return array;
}

#endif
