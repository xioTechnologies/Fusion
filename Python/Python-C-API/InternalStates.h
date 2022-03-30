#ifndef INTERNAL_STATES_H
#define INTERNAL_STATES_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsInternalStates internal_states;
} InternalStates;

static void internal_states_free(InternalStates *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *internal_states_get_acceleration_error(InternalStates *self) {
    return Py_BuildValue("f", self->internal_states.accelerationError);
}

static PyObject *internal_states_get_accelerometer_ignored(InternalStates *self) {
    return build_bool(self->internal_states.accelerometerIgnored);
}

static PyObject *internal_states_get_acceleration_rejection_timer(InternalStates *self) {
    return Py_BuildValue("f", self->internal_states.accelerationRejectionTimer);
}

static PyObject *internal_states_get_magnetic_error(InternalStates *self) {
    return Py_BuildValue("f", self->internal_states.magneticError);
}

static PyObject *internal_states_get_magnetometer_ignored(InternalStates *self) {
    return build_bool(self->internal_states.magnetometerIgnored);
}

static PyObject *internal_states_get_magnetic_rejection_timer(InternalStates *self) {
    return Py_BuildValue("f", self->internal_states.magneticRejectionTimer);
}

static PyGetSetDef internal_states_get_set[] = {
        {"acceleration_error",           (getter) internal_states_get_acceleration_error,           NULL, "", NULL},
        {"accelerometer_ignored",        (getter) internal_states_get_accelerometer_ignored,        NULL, "", NULL},
        {"acceleration_rejection_timer", (getter) internal_states_get_acceleration_rejection_timer, NULL, "", NULL},
        {"magnetic_error",               (getter) internal_states_get_magnetic_error,               NULL, "", NULL},
        {"magnetometer_ignored",         (getter) internal_states_get_magnetometer_ignored,         NULL, "", NULL},
        {"magnetic_rejection_timer",     (getter) internal_states_get_magnetic_rejection_timer,     NULL, "", NULL},
        {NULL}  /* sentinel */
};

static PyTypeObject internal_states_object = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "imufusion.InternalStates",
        .tp_basicsize = sizeof(InternalStates),
        .tp_dealloc = (destructor) internal_states_free,
        .tp_getset = internal_states_get_set,
};

static PyObject *internal_states_from(const FusionAhrsInternalStates *const internal_states) {
    InternalStates *const self = (InternalStates *) internal_states_object.tp_alloc(&internal_states_object, 0);
    self->internal_states = *internal_states;
    return (PyObject *) self;
}

#endif
