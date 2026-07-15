#ifndef BIAS_SETTINGS_H
#define BIAS_SETTINGS_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionBiasSettings wrapped;
} BiasSettings;

static PyObject *bias_settings_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    FusionBiasSettings settings = fusionBiasDefaultSettings;
    int continuous = settings.continuous;

    static char *kwlist[] = {
        "sample_rate",
        "duration",
        "threshold",
        "continuous",
        "holdoff",
        NULL, /* sentinel */
    };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "|fffpf", kwlist,
                                    &settings.sampleRate,
                                    &settings.duration,
                                    &settings.threshold,
                                    &continuous,
                                    &settings.holdoff) == 0) {
        return NULL;
    }

    settings.continuous = continuous != 0;

    BiasSettings *const self = (BiasSettings *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    self->wrapped = settings;
    return (PyObject *) self;
}

static void bias_settings_free(BiasSettings *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *bias_settings_get_sample_rate(BiasSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.sampleRate);
}

static int bias_settings_set_sample_rate(BiasSettings *self, PyObject *value, void *closure) {
    const float sample_rate = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.sampleRate = sample_rate;
    return 0;
}

static PyObject *bias_settings_get_duration(BiasSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.duration);
}

static int bias_settings_set_duration(BiasSettings *self, PyObject *value, void *closure) {
    const float duration = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.duration = duration;
    return 0;
}

static PyObject *bias_settings_get_threshold(BiasSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.threshold);
}

static int bias_settings_set_threshold(BiasSettings *self, PyObject *value, void *closure) {
    const float threshold = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.threshold = threshold;
    return 0;
}

static PyObject *bias_settings_get_continuous(BiasSettings *self) {
    if (self->wrapped.continuous) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static int bias_settings_set_continuous(BiasSettings *self, PyObject *value, void *closure) {
    const int continuous = PyObject_IsTrue(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.continuous = continuous != 0;
    return 0;
}

static PyObject *bias_settings_get_holdoff(BiasSettings *self) {
    return PyFloat_FromDouble((double) self->wrapped.holdoff);
}

static int bias_settings_set_holdoff(BiasSettings *self, PyObject *value, void *closure) {
    const float holdoff = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    self->wrapped.holdoff = holdoff;
    return 0;
}

static PyGetSetDef bias_settings_get_set[] = {
    {"sample_rate", (getter) bias_settings_get_sample_rate, (setter) bias_settings_set_sample_rate, "", NULL},
    {"duration", (getter) bias_settings_get_duration, (setter) bias_settings_set_duration, "", NULL},
    {"threshold", (getter) bias_settings_get_threshold, (setter) bias_settings_set_threshold, "", NULL},
    {"continuous", (getter) bias_settings_get_continuous, (setter) bias_settings_set_continuous, "", NULL},
    {"holdoff", (getter) bias_settings_get_holdoff, (setter) bias_settings_set_holdoff, "", NULL},
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

    self->wrapped = *settings;
    return (PyObject *) self;
}

#endif
