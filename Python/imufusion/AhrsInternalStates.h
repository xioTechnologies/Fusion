#ifndef AHRS_INTERNAL_STATES_H
#define AHRS_INTERNAL_STATES_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsInternalStates internal_states;
} AhrsInternalStates;

static void ahrs_internal_states_free(AhrsInternalStates *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ahrs_internal_states_get_acceleration_error(AhrsInternalStates *self) {
    return PyFloat_FromDouble((double) self->internal_states.accelerationError);
}

static PyObject *ahrs_internal_states_get_accelerometer_ignored(AhrsInternalStates *self) {
    return PyBool_FromLong((long) self->internal_states.accelerometerIgnored);
}

static PyObject *ahrs_internal_states_get_acceleration_recovery_trigger(AhrsInternalStates *self) {
    return PyFloat_FromDouble((double) self->internal_states.accelerationRecoveryTrigger);
}

static PyObject *ahrs_internal_states_get_magnetic_error(AhrsInternalStates *self) {
    return PyFloat_FromDouble((double) self->internal_states.magneticError);
}

static PyObject *ahrs_internal_states_get_magnetometer_ignored(AhrsInternalStates *self) {
    return PyBool_FromLong((long) self->internal_states.magnetometerIgnored);
}

static PyObject *ahrs_internal_states_get_magnetic_recovery_trigger(AhrsInternalStates *self) {
    return PyFloat_FromDouble((double) self->internal_states.magneticRecoveryTrigger);
}

static PyGetSetDef ahrs_internal_states_get_set[] = {
    {"acceleration_error", (getter) ahrs_internal_states_get_acceleration_error, NULL, "", NULL},
    {"accelerometer_ignored", (getter) ahrs_internal_states_get_accelerometer_ignored, NULL, "", NULL},
    {"acceleration_recovery_trigger", (getter) ahrs_internal_states_get_acceleration_recovery_trigger, NULL, "", NULL},
    {"magnetic_error", (getter) ahrs_internal_states_get_magnetic_error, NULL, "", NULL},
    {"magnetometer_ignored", (getter) ahrs_internal_states_get_magnetometer_ignored, NULL, "", NULL},
    {"magnetic_recovery_trigger", (getter) ahrs_internal_states_get_magnetic_recovery_trigger, NULL, "", NULL},
    {NULL} /* sentinel */
};

static PyTypeObject ahrs_internal_states_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.AhrsInternalStates",
    .tp_basicsize = sizeof(AhrsInternalStates),
    .tp_dealloc = (destructor) ahrs_internal_states_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = ahrs_internal_states_get_set,
};

static PyObject *internal_states_from(const FusionAhrsInternalStates *const internal_states) {
    AhrsInternalStates *const self = (AhrsInternalStates *) ahrs_internal_states_object.tp_alloc(&ahrs_internal_states_object, 0);

    if (self == NULL) {
        return NULL;
    }

    self->internal_states = *internal_states;
    return (PyObject *) self;
}

#endif
