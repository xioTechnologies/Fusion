#ifndef AHRS_FLAGS_H
#define AHRS_FLAGS_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsFlags wrapped;
} AhrsFlags;

static void ahrs_flags_free(AhrsFlags *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ahrs_flags_get_initialising(AhrsFlags *self) {
    return PyBool_FromLong((long) self->wrapped.initialising);
}

static PyObject *ahrs_flags_get_angular_rate_recovery(AhrsFlags *self) {
    return PyBool_FromLong((long) self->wrapped.angularRateRecovery);
}

static PyObject *ahrs_flags_get_acceleration_recovery(AhrsFlags *self) {
    return PyBool_FromLong((long) self->wrapped.accelerationRecovery);
}

static PyObject *ahrs_flags_get_magnetic_recovery(AhrsFlags *self) {
    return PyBool_FromLong((long) self->wrapped.magneticRecovery);
}

static PyGetSetDef ahrs_flags_get_set[] = {
    {"initialising", (getter) ahrs_flags_get_initialising, NULL, "", NULL},
    {"angular_rate_recovery", (getter) ahrs_flags_get_angular_rate_recovery, NULL, "", NULL},
    {"acceleration_recovery", (getter) ahrs_flags_get_acceleration_recovery, NULL, "", NULL},
    {"magnetic_recovery", (getter) ahrs_flags_get_magnetic_recovery, NULL, "", NULL},
    {NULL} /* sentinel */
};

static PyTypeObject ahrs_flags_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.AhrsFlags",
    .tp_basicsize = sizeof(AhrsFlags),
    .tp_dealloc = (destructor) ahrs_flags_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = ahrs_flags_get_set,
};

static PyObject *ahrs_flags_from(const FusionAhrsFlags *const flags) {
    AhrsFlags *const self = (AhrsFlags *) ahrs_flags_object.tp_alloc(&ahrs_flags_object, 0);

    if (self == NULL) {
        return NULL;
    }

    self->wrapped = *flags;
    return (PyObject *) self;
}

#endif
