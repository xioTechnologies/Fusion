#ifndef HARD_IRON_SETTINGS_H
#define HARD_IRON_SETTINGS_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionHardIronSettings wrapped;
} HardIronSettings;

static PyObject *hard_iron_settings_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    FusionHardIronSettings settings = fusionHardIronDefaultSettings;

    static char *kwlist[] = {
        "sample_rate",
        "timeout",
        "intensity",
        NULL, /* sentinel */
    };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "|fff", kwlist,
                                    &settings.sampleRate,
                                    &settings.timeout,
                                    &settings.intensity) == 0) {
        return NULL;
    }

    HardIronSettings *const self = (HardIronSettings *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    self->wrapped = settings;
    return (PyObject *) self;
}

static void hard_iron_settings_free(HardIronSettings *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *hard_iron_settings_get_sample_rate(HardIronSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.sampleRate);
}

static int hard_iron_settings_set_sample_rate(HardIronSettings *self, PyObject *value, void *closure) {
    const float sample_rate = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.sampleRate = sample_rate;
    return 0;
}

static PyObject *hard_iron_settings_get_timeout(HardIronSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.timeout);
}

static int hard_iron_settings_set_timeout(HardIronSettings *self, PyObject *value, void *closure) {
    const float timeout = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.timeout = timeout;
    return 0;
}

static PyObject *hard_iron_settings_get_intensity(HardIronSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.intensity);
}

static int hard_iron_settings_set_intensity(HardIronSettings *self, PyObject *value, void *closure) {
    const float intensity = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.intensity = intensity;
    return 0;
}

static PyGetSetDef hard_iron_settings_get_set[] = {
    {"sample_rate", (getter) hard_iron_settings_get_sample_rate, (setter) hard_iron_settings_set_sample_rate, "", NULL},
    {"timeout", (getter) hard_iron_settings_get_timeout, (setter) hard_iron_settings_set_timeout, "", NULL},
    {"intensity", (getter) hard_iron_settings_get_intensity, (setter) hard_iron_settings_set_intensity, "", NULL},
    {NULL} /* sentinel */
};

static PyTypeObject hard_iron_settings_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.HardIronSettings",
    .tp_basicsize = sizeof(HardIronSettings),
    .tp_dealloc = (destructor) hard_iron_settings_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = hard_iron_settings_new,
    .tp_getset = hard_iron_settings_get_set,
};

static PyObject *hard_iron_settings_from(const FusionHardIronSettings *const settings) {
    HardIronSettings *const self = (HardIronSettings *) hard_iron_settings_object.tp_alloc(&hard_iron_settings_object, 0);

    if (self == NULL) {
        return NULL;
    }

    self->wrapped = *settings;
    return (PyObject *) self;
}

#endif
