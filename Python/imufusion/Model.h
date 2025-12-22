#ifndef MODEL_H
#define MODEL_H

#include "../../Fusion/Fusion.h"
#include "NpArray.h"
#include <Python.h>

static PyObject *model_inertial(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *uncalibrated_object = NULL;
    PyObject *misalignment_object = NULL;
    PyObject *sensitivity_object = NULL;
    PyObject *offset_object = NULL;

    static char *kwlist[] = {
        "uncalibrated",
        "misalignment",
        "sensitivity",
        "offset",
        NULL, /* sentinel */
    };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "O|OOO", kwlist,
                                    &uncalibrated_object,
                                    &misalignment_object,
                                    &sensitivity_object,
                                    &offset_object) == 0) {
        return NULL;
    }

    FusionVector uncalibrated;

    if (np_array_1x3_to(uncalibrated.array, uncalibrated_object) != 0) {
        return NULL;
    }

    FusionMatrix misalignment = FUSION_MATRIX_IDENTITY;

    if ((misalignment_object != NULL) && (np_array_3x3_to(misalignment.array, misalignment_object) != 0)) {
        return NULL;
    }

    FusionVector sensitivity = FUSION_VECTOR_ONES;

    if ((sensitivity_object != NULL) && (np_array_1x3_to(sensitivity.array, sensitivity_object) != 0)) {
        return NULL;
    }

    FusionVector offset = FUSION_VECTOR_ZERO;

    if ((offset_object != NULL) && (np_array_1x3_to(offset.array, offset_object) != 0)) {
        return NULL;
    }

    const FusionVector calibrated = FusionModelInertial(uncalibrated, misalignment, sensitivity, offset);

    return np_array_1x3_from(calibrated.array);
}

static PyObject *model_magnetic(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *uncalibrated_object = NULL;
    PyObject *soft_iron_matrix_object = NULL;
    PyObject *hard_iron_offset_object = NULL;

    static char *kwlist[] = {
        "uncalibrated",
        "soft_iron_matrix",
        "hard_iron_offset",
        NULL, /* sentinel */
    };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist,
                                    &uncalibrated_object,
                                    &soft_iron_matrix_object,
                                    &hard_iron_offset_object) == 0) {
        return NULL;
    }

    FusionVector uncalibrated;

    if (np_array_1x3_to(uncalibrated.array, uncalibrated_object) != 0) {
        return NULL;
    }

    FusionMatrix soft_iron_matrix = FUSION_MATRIX_IDENTITY;

    if ((soft_iron_matrix_object != NULL) && (np_array_3x3_to(soft_iron_matrix.array, soft_iron_matrix_object) != 0)) {
        return NULL;
    }

    FusionVector hard_iron_offset = FUSION_VECTOR_ZERO;

    if ((hard_iron_offset_object != NULL) && (np_array_1x3_to(hard_iron_offset.array, hard_iron_offset_object) != 0)) {
        return NULL;
    }

    const FusionVector calibrated = FusionModelMagnetic(uncalibrated, soft_iron_matrix, hard_iron_offset);

    return np_array_1x3_from(calibrated.array);
}

static PyMethodDef model_methods[] = {
    {"model_inertial", (PyCFunction) model_inertial, METH_VARARGS | METH_KEYWORDS, ""},
    {"model_magnetic", (PyCFunction) model_magnetic, METH_VARARGS | METH_KEYWORDS, ""},
    {NULL} /* sentinel */
};

#endif
