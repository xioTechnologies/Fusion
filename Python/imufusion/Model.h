#ifndef MODEL_H
#define MODEL_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

static PyObject *model_inertial(PyObject *self, PyObject *args) {
    PyObject *uncalibrated_object;
    PyObject *misalignment_object;
    PyObject *sensitivity_object;
    PyObject *offset_object;

    if (PyArg_ParseTuple(args, "OOOO", &uncalibrated_object, &misalignment_object, &sensitivity_object, &offset_object) == 0) {
        return NULL;
    }

    FusionVector uncalibrated;

    if (np_array_1x3_to(uncalibrated.array, uncalibrated_object) != 0) {
        return NULL;
    }

    FusionMatrix misalignment;

    if (np_array_3x3_to(misalignment.array, misalignment_object) != 0) {
        return NULL;
    }

    FusionVector sensitivity;

    if (np_array_1x3_to(sensitivity.array, sensitivity_object) != 0) {
        return NULL;
    }

    FusionVector offset;

    if (np_array_1x3_to(offset.array, offset_object) != 0) {
        return NULL;
    }

    const FusionVector calibrated = FusionModelInertial(uncalibrated, misalignment, sensitivity, offset);

    return np_array_1x3_from(calibrated.array);
}

static PyObject *model_magnetic(PyObject *self, PyObject *args) {
    PyObject *uncalibrated_object;
    PyObject *soft_iron_matrix_object;
    PyObject *hard_iron_offset_object;

    if (PyArg_ParseTuple(args, "OOO", &uncalibrated_object, &soft_iron_matrix_object, &hard_iron_offset_object) == 0) {
        return NULL;
    }

    FusionVector uncalibrated;

    if (np_array_1x3_to(uncalibrated.array, uncalibrated_object) != 0) {
        return NULL;
    }

    FusionMatrix soft_iron_matrix;

    if (np_array_3x3_to(soft_iron_matrix.array, soft_iron_matrix_object) != 0) {
        return NULL;
    }

    FusionVector hard_iron_offset;

    if (np_array_1x3_to(hard_iron_offset.array, hard_iron_offset_object) != 0) {
        return NULL;
    }

    const FusionVector calibrated = FusionModelMagnetic(uncalibrated, soft_iron_matrix, hard_iron_offset);

    return np_array_1x3_from(calibrated.array);
}

static PyMethodDef model_methods[] = {
    {"model_inertial", (PyCFunction) model_inertial, METH_VARARGS, ""},
    {"model_magnetic", (PyCFunction) model_magnetic, METH_VARARGS, ""},
    {NULL} /* sentinel */
};

#endif
