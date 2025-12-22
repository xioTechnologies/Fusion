#ifndef AHRS_SETTINGS_H
#define AHRS_SETTINGS_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsSettings settings;
} AhrsSettings;

static PyObject *ahrs_settings_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    AhrsSettings *const self = (AhrsSettings *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    if (PyArg_ParseTuple(args, "iffffI",
                         &self->settings.convention,
                         &self->settings.gain,
                         &self->settings.gyroscopeRange,
                         &self->settings.accelerationRejection,
                         &self->settings.magneticRejection,
                         &self->settings.recoveryTriggerPeriod) == 0) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *) self;
}

static void ahrs_settings_free(AhrsSettings *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ahrs_settings_get_convention(AhrsSettings *self) {
    return PyLong_FromLong(self->settings.convention);
}

static int ahrs_settings_set_convention(AhrsSettings *self, PyObject *value, void *closure) {
    const FusionConvention convention = (FusionConvention) PyLong_AsUnsignedLong(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.convention = convention;
    return 0;
}

static PyObject *ahrs_settings_get_gain(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->settings.gain);
}

static int ahrs_settings_set_gain(AhrsSettings *self, PyObject *value, void *closure) {
    const float gain = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.gain = gain;
    return 0;
}

static PyObject *ahrs_settings_get_gyroscope_range(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->settings.gyroscopeRange);
}

static int ahrs_settings_set_gyroscope_range(AhrsSettings *self, PyObject *value, void *closure) {
    const float gyroscope_range = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.gyroscopeRange = gyroscope_range;
    return 0;
}

static PyObject *ahrs_settings_get_acceleration_rejection(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->settings.accelerationRejection);
}

static int ahrs_settings_set_acceleration_rejection(AhrsSettings *self, PyObject *value, void *closure) {
    const float acceleration_rejection = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.accelerationRejection = acceleration_rejection;
    return 0;
}

static PyObject *ahrs_settings_get_magnetic_rejection(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->settings.magneticRejection);
}

static int ahrs_settings_set_magnetic_rejection(AhrsSettings *self, PyObject *value, void *closure) {
    const float magnetic_rejection = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.magneticRejection = magnetic_rejection;
    return 0;
}

static PyObject *ahrs_settings_get_recovery_trigger_period(AhrsSettings *self) {
    return PyLong_FromUnsignedLong((unsigned long) self->settings.recoveryTriggerPeriod);
}

static int ahrs_settings_set_recovery_trigger_period(AhrsSettings *self, PyObject *value, void *closure) {
    const unsigned int recovery_trigger_period = (unsigned int) PyLong_AsUnsignedLong(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.recoveryTriggerPeriod = recovery_trigger_period;
    return 0;
}

static PyGetSetDef ahrs_settings_get_set[] = {
    {"convention", (getter) ahrs_settings_get_convention, (setter) ahrs_settings_set_convention, "", NULL},
    {"gain", (getter) ahrs_settings_get_gain, (setter) ahrs_settings_set_gain, "", NULL},
    {"gyroscope_range", (getter) ahrs_settings_get_gyroscope_range, (setter) ahrs_settings_set_gyroscope_range, "", NULL},
    {"acceleration_rejection", (getter) ahrs_settings_get_acceleration_rejection, (setter) ahrs_settings_set_acceleration_rejection, "", NULL},
    {"magnetic_rejection", (getter) ahrs_settings_get_magnetic_rejection, (setter) ahrs_settings_set_magnetic_rejection, "", NULL},
    {"recovery_trigger_period", (getter) ahrs_settings_get_recovery_trigger_period, (setter) ahrs_settings_set_recovery_trigger_period, "", NULL},
    {NULL} /* sentinel */
};

static PyTypeObject ahrs_settings_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.AhrsSettings",
    .tp_basicsize = sizeof(AhrsSettings),
    .tp_dealloc = (destructor) ahrs_settings_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ahrs_settings_new,
    .tp_getset = ahrs_settings_get_set,
};

#endif
