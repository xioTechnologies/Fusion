#ifndef REMAP_H
#define REMAP_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

static PyObject *remap(PyObject *self, PyObject *args) {
    PyObject *sensor_object;
    int alignment;

    if (PyArg_ParseTuple(args, "Oi", &sensor_object, &alignment) == 0) {
        return NULL;
    }

    FusionVector sensor;

    if (np_array_1x3_to(sensor.array, sensor_object) != 0) {
        return NULL;
    }

    const FusionVector remapped_sensor = FusionRemap(sensor, (FusionRemapAlignment) alignment);

    return np_array_1x3_from(remapped_sensor.array);
}

static PyMethodDef remap_methods[] = {
    {"remap", (PyCFunction) remap, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
