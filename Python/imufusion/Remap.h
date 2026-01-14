#ifndef REMAP_H
#define REMAP_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

static int remap_alignment_from(FusionRemapAlignment *const alignment, const int alignment_int) {
    switch (alignment_int) {
        case FusionRemapAlignmentPXPYPZ:
        case FusionRemapAlignmentPXPZNY:
        case FusionRemapAlignmentPXNZPY:
        case FusionRemapAlignmentPXNYNZ:
        case FusionRemapAlignmentPYPXNZ:
        case FusionRemapAlignmentPYPZPX:
        case FusionRemapAlignmentPYNZNX:
        case FusionRemapAlignmentPYNXPZ:
        case FusionRemapAlignmentPZPXPY:
        case FusionRemapAlignmentPZPYNX:
        case FusionRemapAlignmentPZNYPX:
        case FusionRemapAlignmentPZNXNY:
        case FusionRemapAlignmentNZPXNY:
        case FusionRemapAlignmentNZPYPX:
        case FusionRemapAlignmentNZNYNX:
        case FusionRemapAlignmentNZNXPY:
        case FusionRemapAlignmentNYPXPZ:
        case FusionRemapAlignmentNYPZNX:
        case FusionRemapAlignmentNYNZPX:
        case FusionRemapAlignmentNYNXNZ:
        case FusionRemapAlignmentNXPYNZ:
        case FusionRemapAlignmentNXPZPY:
        case FusionRemapAlignmentNXNZNY:
        case FusionRemapAlignmentNXNYPZ:
            *alignment = (FusionRemapAlignment) alignment_int;
            return 0;
    }

    PyErr_SetString(PyExc_ValueError, "'alignment' must be imufusion.ALIGNMENT_*");
    return -1;
}

static PyObject *remap(PyObject *self, PyObject *args) {
    PyObject *sensor_object;
    int alignment_int;

    if (PyArg_ParseTuple(args, "Oi", &sensor_object, &alignment_int) == 0) {
        return NULL;
    }

    FusionVector sensor;

    if (np_array_1x3_to(sensor.array, sensor_object) != 0) {
        return NULL;
    }

    FusionRemapAlignment alignment;

    if (remap_alignment_from(&alignment, alignment_int) != 0) {
        return NULL;
    }

    const FusionVector remapped_sensor = FusionRemap(sensor, alignment);

    return np_array_1x3_from(remapped_sensor.array);
}

static PyMethodDef remap_methods[] = {
    {"remap", (PyCFunction) remap, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
