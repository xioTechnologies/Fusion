#ifndef AHRS_SETTINGS_H
#define AHRS_SETTINGS_H

#include "../../Fusion/Fusion.h"
#include "Convention.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsSettings wrapped;
} AhrsSettings;

static PyObject *ahrs_settings_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    int convention_int = fusionAhrsDefaultSettings.convention;
    FusionAhrsSettings settings = fusionAhrsDefaultSettings;

    static char *kwlist[] = {
        "convention",
        "gain",
        "gyroscope_range",
        "acceleration_rejection",
        "magnetic_rejection",
        "recovery_trigger_period",
        NULL, /* sentinel */
    };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "|iffffI", kwlist,
                                    &convention_int,
                                    &settings.gain,
                                    &settings.gyroscopeRange,
                                    &settings.accelerationRejection,
                                    &settings.magneticRejection,
                                    &settings.recoveryTriggerPeriod) == 0) {
        return NULL;
    }

    FusionConvention convention;

    if (convention_from(&convention, convention_int) != 0) {
        return NULL;
    }

    AhrsSettings *const self = (AhrsSettings *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    self->wrapped = settings;
    return (PyObject *) self;
}

static void ahrs_settings_free(AhrsSettings *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ahrs_settings_get_convention(AhrsSettings *self) {
    return PyLong_FromLong(self->wrapped.convention);
}

static int ahrs_settings_set_convention(AhrsSettings *self, PyObject *value, void *closure) {
    const int convention_int = (int) PyLong_AsUnsignedLong(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    FusionConvention convention;

    if (convention_from(&convention, convention_int) != 0) {
        return -1;
    }

    self->wrapped.convention = convention;
    return 0;
}

static PyObject *ahrs_settings_get_gain(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.gain);
}

static int ahrs_settings_set_gain(AhrsSettings *self, PyObject *value, void *closure) {
    const float gain = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.gain = gain;
    return 0;
}

static PyObject *ahrs_settings_get_gyroscope_range(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.gyroscopeRange);
}

static int ahrs_settings_set_gyroscope_range(AhrsSettings *self, PyObject *value, void *closure) {
    const float gyroscope_range = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.gyroscopeRange = gyroscope_range;
    return 0;
}

static PyObject *ahrs_settings_get_acceleration_rejection(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.accelerationRejection);
}

static int ahrs_settings_set_acceleration_rejection(AhrsSettings *self, PyObject *value, void *closure) {
    const float acceleration_rejection = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.accelerationRejection = acceleration_rejection;
    return 0;
}

static PyObject *ahrs_settings_get_magnetic_rejection(AhrsSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.magneticRejection);
}

static int ahrs_settings_set_magnetic_rejection(AhrsSettings *self, PyObject *value, void *closure) {
    const float magnetic_rejection = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.magneticRejection = magnetic_rejection;
    return 0;
}

static PyObject *ahrs_settings_get_recovery_trigger_period(AhrsSettings *self) {
    return PyLong_FromUnsignedLong((unsigned long) self->wrapped.recoveryTriggerPeriod);
}

static int ahrs_settings_set_recovery_trigger_period(AhrsSettings *self, PyObject *value, void *closure) {
    const unsigned int recovery_trigger_period = (unsigned int) PyLong_AsUnsignedLong(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.recoveryTriggerPeriod = recovery_trigger_period;
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
