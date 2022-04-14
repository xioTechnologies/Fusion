#ifndef FLAGS_H
#define FLAGS_H

#include "../../Fusion/Fusion.h"
#include "Helpers.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrsFlags flags;
} Flags;

static void flags_free(Flags *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *flags_get_initialising(Flags *self) {
    return build_bool(self->flags.initialising);
}

static PyObject *flags_get_acceleration_rejection_warning(Flags *self) {
    return build_bool(self->flags.accelerationRejectionWarning);
}

static PyObject *flags_get_acceleration_rejection_timeout(Flags *self) {
    return build_bool(self->flags.accelerationRejectionTimeout);
}

static PyObject *flags_get_magnetic_rejection_warning(Flags *self) {
    return build_bool(self->flags.magneticRejectionWarning);
}

static PyObject *flags_get_magnetic_rejection_timeout(Flags *self) {
    return build_bool(self->flags.magneticRejectionTimeout);
}

static PyGetSetDef flags_get_set[] = {
        {"initialising",                   (getter) flags_get_initialising,                   NULL, "", NULL},
        {"acceleration_rejection_warning", (getter) flags_get_acceleration_rejection_warning, NULL, "", NULL},
        {"acceleration_rejection_timeout", (getter) flags_get_acceleration_rejection_timeout, NULL, "", NULL},
        {"magnetic_rejection_warning",     (getter) flags_get_magnetic_rejection_warning,     NULL, "", NULL},
        {"magnetic_rejection_timeout",     (getter) flags_get_magnetic_rejection_timeout,     NULL, "", NULL},
        {NULL}  /* sentinel */
};

static PyTypeObject flags_object = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "imufusion.Flags",
        .tp_basicsize = sizeof(Flags),
        .tp_dealloc = (destructor) flags_free,
        .tp_getset = flags_get_set,
};

static PyObject *flags_from(const FusionAhrsFlags *const flags) {
    Flags *const self = (Flags *) flags_object.tp_alloc(&flags_object, 0);
    self->flags = *flags;
    return (PyObject *) self;
}

#endif
