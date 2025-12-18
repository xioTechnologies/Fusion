#ifndef AXES_H
#define AXES_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

static PyObject *axes_swap(PyObject *self, PyObject *args) {
    PyObject *sensor_object;
    int alignment;

    if (PyArg_ParseTuple(args, "Oi", &sensor_object, &alignment) == 0) {
        return NULL;
    }

    FusionVector sensor;

    if (np_array_1x3_to(sensor.array, sensor_object) != 0) {
        return NULL;
    }

    const FusionVector aligned_sensor = FusionAxesSwap(sensor, (FusionAxesAlignment) alignment);

    return np_array_1x3_from(aligned_sensor.array);
}

static PyMethodDef axes_methods[] = {
    {"axes_swap", (PyCFunction) axes_swap, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
