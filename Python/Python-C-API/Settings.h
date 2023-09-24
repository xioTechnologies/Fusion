#ifndef SETTINGS_H
#define SETTINGS_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsSettings settings;
} Settings;

static PyObject *settings_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    Settings *const self = (Settings *) subtype->tp_alloc(subtype, 0);

    const char *const error = PARSE_TUPLE(args, "iffffI", &self->settings.convention, &self->settings.gain, &self->settings.gyroscopeRange, &self->settings.accelerationRejection, &self->settings.magneticRejection, &self->settings.recoveryTriggerPeriod);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }
    return (PyObject *) self;
}

static void settings_free(Settings *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *settings_get_convention(Settings *self) {
    return Py_BuildValue("l", self->settings.convention);
}

static int settings_set_convention(Settings *self, PyObject *value, void *closure) {
    const FusionConvention convention = (FusionConvention) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.convention = convention;
    return 0;
}

static PyObject *settings_get_gain(Settings *self) {
    return Py_BuildValue("f", self->settings.gain);
}

static int settings_set_gain(Settings *self, PyObject *value, void *closure) {
    const float gain = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.gain = gain;
    return 0;
}

static PyObject *settings_get_gyroscope_range(Settings *self) {
    return Py_BuildValue("f", self->settings.gyroscopeRange);
}

static int settings_set_gyroscope_range(Settings *self, PyObject *value, void *closure) {
    const float gyroscope_range = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.gyroscopeRange = gyroscope_range;
    return 0;
}

static PyObject *settings_get_acceleration_rejection(Settings *self) {
    return Py_BuildValue("f", self->settings.accelerationRejection);
}

static int settings_set_acceleration_rejection(Settings *self, PyObject *value, void *closure) {
    const float acceleration_rejection = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.accelerationRejection = acceleration_rejection;
    return 0;
}

static PyObject *settings_get_magnetic_rejection(Settings *self) {
    return Py_BuildValue("f", self->settings.magneticRejection);
}

static int settings_set_magnetic_rejection(Settings *self, PyObject *value, void *closure) {
    const float magnetic_rejection = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.magneticRejection = magnetic_rejection;
    return 0;
}

static PyObject *settings_get_recovery_trigger_period(Settings *self) {
    return Py_BuildValue("I", self->settings.recoveryTriggerPeriod);
}

static int settings_set_recovery_trigger_period(Settings *self, PyObject *value, void *closure) {
    const unsigned int recovery_trigger_period = (unsigned int) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.recoveryTriggerPeriod = recovery_trigger_period;
    return 0;
}

static PyGetSetDef settings_get_set[] = {
        {"convention",              (getter) settings_get_convention,              (setter) settings_set_convention,              "", NULL},
        {"gain",                    (getter) settings_get_gain,                    (setter) settings_set_gain,                    "", NULL},
        {"gyroscope_range",         (getter) settings_get_gyroscope_range,         (setter) settings_set_gyroscope_range,         "", NULL},
        {"acceleration_rejection",  (getter) settings_get_acceleration_rejection,  (setter) settings_set_acceleration_rejection,  "", NULL},
        {"magnetic_rejection",      (getter) settings_get_magnetic_rejection,      (setter) settings_set_magnetic_rejection,      "", NULL},
        {"recovery_trigger_period", (getter) settings_get_recovery_trigger_period, (setter) settings_set_recovery_trigger_period, "", NULL},
        {NULL}  /* sentinel */
};

static PyTypeObject settings_object = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "imufusion.Settings",
        .tp_basicsize = sizeof(Settings),
        .tp_dealloc = (destructor) settings_free,
        .tp_new = settings_new,
        .tp_getset = settings_get_set,
};

#endif
