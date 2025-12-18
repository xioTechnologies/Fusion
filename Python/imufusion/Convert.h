#ifndef CONVERT_H
#define CONVERT_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

static PyObject *quaternion_to_matrix(PyObject *self, PyObject *args) {
    PyObject *quaternion_object;

    if (PyArg_ParseTuple(args, "O", &quaternion_object) == 0) {
        return NULL;
    }

    FusionQuaternion quaternion;

    if (np_array_1x4_to(quaternion.array, quaternion_object) != 0) {
        return NULL;
    }

    const FusionMatrix matrix = FusionQuaternionToMatrix(quaternion);

    return np_array_3x3_from(matrix.array);
}

static PyObject *quaternion_to_euler(PyObject *self, PyObject *args) {
    PyObject *quaternion_object;

    if (PyArg_ParseTuple(args, "O", &quaternion_object) == 0) {
        return NULL;
    }

    FusionQuaternion quaternion;

    if (np_array_1x4_to(quaternion.array, quaternion_object) != 0) {
        return NULL;
    }

    const FusionEuler euler = FusionQuaternionToEuler(quaternion);

    return np_array_1x3_from(euler.array);
}

static PyMethodDef convert_methods[] = {
    {"quaternion_to_matrix", (PyCFunction) quaternion_to_matrix, METH_VARARGS, ""},
    {"quaternion_to_euler", (PyCFunction) quaternion_to_euler, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
