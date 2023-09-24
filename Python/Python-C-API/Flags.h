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

static PyObject *flags_get_angular_rate_recovery(Flags *self) {
    return build_bool(self->flags.angularRateRecovery);
}

static PyObject *flags_get_acceleration_recovery(Flags *self) {
    return build_bool(self->flags.accelerationRecovery);
}

static PyObject *flags_get_magnetic_recovery(Flags *self) {
    return build_bool(self->flags.magneticRecovery);
}

static PyGetSetDef flags_get_set[] = {
        {"initialising",          (getter) flags_get_initialising,          NULL, "", NULL},
        {"angular_rate_recovery", (getter) flags_get_angular_rate_recovery, NULL, "", NULL},
        {"acceleration_recovery", (getter) flags_get_acceleration_recovery, NULL, "", NULL},
        {"magnetic_recovery",     (getter) flags_get_magnetic_recovery,     NULL, "", NULL},
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
