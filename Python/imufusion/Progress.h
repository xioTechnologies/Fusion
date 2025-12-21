#ifndef PROGRESS_H
#define PROGRESS_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionProgress wrapped;
} Progress;

static void progress_free(Progress *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *progress_get_status(Progress *self) {
    return PyLong_FromLong((long) self->wrapped.status);
}

static PyObject *progress_get_percentage(Progress *self) {
    return PyLong_FromUnsignedLong((unsigned long) self->wrapped.percentage);
}

static PyGetSetDef progress_get_set[] = {
    {"status", (getter) progress_get_status, NULL, "", NULL},
    {"percentage", (getter) progress_get_percentage, NULL, "", NULL},
    {NULL} /* sentinel */
};

static PyTypeObject progress_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.Progress",
    .tp_basicsize = sizeof(Progress),
    .tp_dealloc = (destructor) progress_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = progress_get_set,
};

static PyObject *progress_from(const FusionProgress *const progress) {
    Progress *const self = (Progress *) progress_object.tp_alloc(&progress_object, 0);

    if (self == NULL) {
        return NULL;
    }

    self->wrapped = *progress;
    return (PyObject *) self;
}

static PyObject *progress_status_to_string(PyObject *null, PyObject *arg) {
    const long status = PyLong_AsLong(arg);

    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyUnicode_FromString(FusionProgressStatusToString((FusionProgressStatus) status));
}

static PyMethodDef progress_methods[] = {
    {"progress_status_to_string", (PyCFunction) progress_status_to_string, METH_O, ""},
    {NULL} /* sentinel */
};

#endif
