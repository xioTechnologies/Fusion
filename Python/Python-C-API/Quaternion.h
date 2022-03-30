#ifndef QUATERNION_H
#define QUATERNION_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    FusionQuaternion quaternion;
} Quaternion;

static PyObject *quaternion_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    PyArrayObject *array;

    const char *error = PARSE_TUPLE(args, "O!", &PyArray_Type, &array);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionQuaternion quaternion;

    error = parse_array(quaternion.array, array, 4);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    Quaternion *const self = (Quaternion *) subtype->tp_alloc(subtype, 0);
    self->quaternion = quaternion;
    return (PyObject *) self;
}

static void quaternion_free(Quaternion *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *quaternion_get_array(Quaternion *self) {
    const npy_intp dims[] = {4};
    return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, self->quaternion.array);
}

static PyObject *quaternion_get_w(Quaternion *self) {
    return Py_BuildValue("f", self->quaternion.element.w);
}

static PyObject *quaternion_get_x(Quaternion *self) {
    return Py_BuildValue("f", self->quaternion.element.x);
}

static PyObject *quaternion_get_y(Quaternion *self) {
    return Py_BuildValue("f", self->quaternion.element.y);
}

static PyObject *quaternion_get_z(Quaternion *self) {
    return Py_BuildValue("f", self->quaternion.element.z);
}

static PyObject *quaternion_to_matrix(Quaternion *self, PyObject *args) {
    FusionMatrix *const matrix = malloc(sizeof(FusionMatrix));
    *matrix = FusionQuaternionToMatrix(self->quaternion);

    const npy_intp dims[] = {3, 3};
    PyObject *array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, matrix->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyObject *quaternion_to_euler(Quaternion *self, PyObject *args) {
    FusionEuler *const euler = malloc(sizeof(FusionEuler));
    *euler = FusionQuaternionToEuler(self->quaternion);

    const npy_intp dims[] = {3};
    PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, euler->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyGetSetDef quaternion_get_set[] = {
        {"array", (getter) quaternion_get_array, NULL, "", NULL},
        {"w",     (getter) quaternion_get_w,     NULL, "", NULL},
        {"x",     (getter) quaternion_get_x,     NULL, "", NULL},
        {"y",     (getter) quaternion_get_y,     NULL, "", NULL},
        {"z",     (getter) quaternion_get_z,     NULL, "", NULL},
        {NULL}  /* sentinel */
};

static PyMethodDef quaternion_methods[] = {
        {"to_matrix", (PyCFunction) quaternion_to_matrix, METH_NOARGS, ""},
        {"to_euler",  (PyCFunction) quaternion_to_euler,  METH_NOARGS, ""},
        {NULL} /* sentinel */
};

static PyTypeObject quaternion_object = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "imufusion.Quaternion",
        .tp_basicsize = sizeof(Quaternion),
        .tp_dealloc = (destructor) quaternion_free,
        .tp_new = quaternion_new,
        .tp_getset = quaternion_get_set,
        .tp_methods = quaternion_methods,
};

static PyObject *quaternion_from(const FusionQuaternion *const quaternion) {
    Quaternion *const self = (Quaternion *) quaternion_object.tp_alloc(&quaternion_object, 0);
    self->quaternion = *quaternion;
    return (PyObject *) self;
}

#endif
