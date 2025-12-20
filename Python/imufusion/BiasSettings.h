#ifndef BIAS_SETTINGS_H
#define BIAS_SETTINGS_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionBiasSettings settings;
} BiasSettings;

static PyObject *bias_settings_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    FusionBiasSettings settings = fusionBiasDefaultSettings;

    static char *kwlist[] = {
        "sample_rate",
        "stationary_period",
        "stationary_threshold",
        NULL, /* sentinel */
    };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "|fff", kwlist,
                                    &settings.sampleRate,
                                    &settings.stationaryPeriod,
                                    &settings.stationaryThreshold) == 0) {
        return NULL;
    }

    BiasSettings *const self = (BiasSettings *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    self->settings = settings;
    return (PyObject *) self;
}

static void bias_settings_free(BiasSettings *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *bias_settings_get_sample_rate(BiasSettings *self) {
    return PyFloat_FromDouble(self->settings.sampleRate);
}

static int bias_settings_set_sample_rate(BiasSettings *self, PyObject *value, void *closure) {
    const float sample_rate = (float) PyLong_AsUnsignedLong(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.sampleRate = sample_rate;
    return 0;
}

static PyObject *bias_settings_get_stationary_period(BiasSettings *self) {
    return PyFloat_FromDouble((double) self->settings.stationaryPeriod);
}

static int bias_settings_set_stationary_period(BiasSettings *self, PyObject *value, void *closure) {
    const float stationary_period = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.stationaryPeriod = stationary_period;
    return 0;
}

static PyObject *bias_settings_get_stationary_threshold(BiasSettings *self) {
    return PyFloat_FromDouble((double) self->settings.stationaryThreshold);
}

static int bias_settings_set_stationary_threshold(BiasSettings *self, PyObject *value, void *closure) {
    const float stationary_threshold = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->settings.stationaryThreshold = stationary_threshold;
    return 0;
}

static PyGetSetDef bias_settings_get_set[] = {
    {"sample_rate", (getter) bias_settings_get_sample_rate, (setter) bias_settings_set_sample_rate, "", NULL},
    {"stationary_period", (getter) bias_settings_get_stationary_period, (setter) bias_settings_set_stationary_period, "", NULL},
    {"stationary_threshold", (getter) bias_settings_get_stationary_threshold, (setter) bias_settings_set_stationary_threshold, "", NULL},
    {NULL} /* sentinel */
};

static PyTypeObject bias_settings_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.BiasSettings",
    .tp_basicsize = sizeof(BiasSettings),
    .tp_dealloc = (destructor) bias_settings_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = bias_settings_new,
    .tp_getset = bias_settings_get_set,
};

static PyObject *bias_settings_from(const FusionBiasSettings *const settings) {
    BiasSettings *const self = (BiasSettings *) bias_settings_object.tp_alloc(&bias_settings_object, 0);

    if (self == NULL) {
        return NULL;
    }

    self->settings = *settings;
    return (PyObject *) self;
}

#endif
