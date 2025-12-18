#ifndef QUATERNION_H
#define QUATERNION_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionQuaternion quaternion;
} Quaternion;

static PyObject *quaternion_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    PyObject *quaternion_object;

    if (PyArg_ParseTuple(args, "O", &quaternion_object) == 0) {
        return NULL;
    }

    FusionQuaternion quaternion;

    if (np_array_1x4_to(quaternion.array, quaternion_object) != 0) {
        return NULL;
    }

    Quaternion *const self = (Quaternion *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    self->quaternion = quaternion;
    return (PyObject *) self;
}

static void quaternion_free(Quaternion *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *quaternion_get_wxyz(Quaternion *self) {
    return np_array_1x4_from(self->quaternion.array);
}

static PyObject *quaternion_get_w(Quaternion *self) {
    return PyFloat_FromDouble((double) self->quaternion.element.w);
}

static PyObject *quaternion_get_x(Quaternion *self) {
    return PyFloat_FromDouble((double) self->quaternion.element.x);
}

static PyObject *quaternion_get_y(Quaternion *self) {
    return PyFloat_FromDouble((double) self->quaternion.element.y);
}

static PyObject *quaternion_get_z(Quaternion *self) {
    return PyFloat_FromDouble((double) self->quaternion.element.z);
}

static PyObject *quaternion_to_matrix_(Quaternion *self, PyObject *args) {
    const FusionMatrix matrix = FusionQuaternionToMatrix(self->quaternion);

    return np_array_3x3_from(matrix.array);
}

static PyObject *quaternion_to_euler_(Quaternion *self, PyObject *args) {
    const FusionEuler euler = FusionQuaternionToEuler(self->quaternion);

    return np_array_1x3_from(euler.array);
}

static PyGetSetDef quaternion_get_set[] = {
    {"wxyz", (getter) quaternion_get_wxyz, NULL, "", NULL},
    {"w", (getter) quaternion_get_w, NULL, "", NULL},
    {"x", (getter) quaternion_get_x, NULL, "", NULL},
    {"y", (getter) quaternion_get_y, NULL, "", NULL},
    {"z", (getter) quaternion_get_z, NULL, "", NULL},
    {NULL} /* sentinel */
};

static PyMethodDef quaternion_methods[] = {
    {"to_matrix", (PyCFunction) quaternion_to_matrix_, METH_NOARGS, ""},
    {"to_euler", (PyCFunction) quaternion_to_euler_, METH_NOARGS, ""},
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

    if (self == NULL) {
        return NULL;
    }

    self->quaternion = *quaternion;
    return (PyObject *) self;
}

#endif
