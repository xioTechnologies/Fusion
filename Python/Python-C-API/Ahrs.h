#ifndef AHRS_H
#define AHRS_H

#include "../../Fusion/Fusion.h"
#include "Flags.h"
#include "Helpers.h"
#include "InternalStates.h"
#include <Python.h>
#include "Quaternion.h"
#include "Settings.h"
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    FusionAhrs ahrs;
} Ahrs;

static PyObject *ahrs_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    Ahrs *const self = (Ahrs *) subtype->tp_alloc(subtype, 0);
    FusionAhrsInitialise(&self->ahrs);
    return (PyObject *) self;
}

static void ahrs_free(Ahrs *self) {
    Py_TYPE(self)->tp_free(self);
}

static int ahrs_set_settings(Ahrs *self, PyObject *value, void *closure) {
    if (PyObject_IsInstance(value, (PyObject *) &settings_object) == false) {
        static char string[64];
        snprintf(string, sizeof(string), "Value type is not %s", settings_object.tp_name);
        PyErr_SetString(PyExc_TypeError, string);
        return -1;
    }
    FusionAhrsSetSettings(&self->ahrs, &((Settings *) value)->settings);
    return 0;
}

static PyObject *ahrs_get_quaternion(Ahrs *self) {
    const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&self->ahrs);
    return quaternion_from(&quaternion);
}

static PyObject *ahrs_get_linear_acceleration(Ahrs *self) {
    FusionVector *const linear_acceleration = malloc(sizeof(FusionVector));
    *linear_acceleration = FusionAhrsGetLinearAcceleration(&self->ahrs);

    const npy_intp dims[] = {3};
    PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, linear_acceleration->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyObject *ahrs_get_earth_acceleration(Ahrs *self) {
    FusionVector *const earth_acceleration = malloc(sizeof(FusionVector));
    *earth_acceleration = FusionAhrsGetEarthAcceleration(&self->ahrs);

    const npy_intp dims[] = {3};
    PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, earth_acceleration->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
}

static PyObject *ahrs_get_internal_states(Ahrs *self) {
    const FusionAhrsInternalStates internal_states = FusionAhrsGetInternalStates(&self->ahrs);
    return internal_states_from(&internal_states);
}

static PyObject *ahrs_get_flags(Ahrs *self) {
    const FusionAhrsFlags flags = FusionAhrsGetFlags(&self->ahrs);
    return flags_from(&flags);
}

static PyObject *ahrs_reset(Ahrs *self, PyObject *args) {
    FusionAhrsReset(&self->ahrs);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ahrs_update(Ahrs *self, PyObject *args) {
    PyArrayObject *gyroscope_array;
    PyArrayObject *accelerometer_array;
    PyArrayObject *magnetometer_array;
    float delta_time;

    const char *error = PARSE_TUPLE(args, "O!O!O!f", &PyArray_Type, &gyroscope_array, &PyArray_Type, &accelerometer_array, &PyArray_Type, &magnetometer_array, &delta_time);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector gyroscope_vector;
    FusionVector accelerometer_vector;
    FusionVector magnetometer_vector;

    error = parse_array(gyroscope_vector.array, gyroscope_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    error = parse_array(accelerometer_vector.array, accelerometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    error = parse_array(magnetometer_vector.array, magnetometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionAhrsUpdate(&self->ahrs, gyroscope_vector, accelerometer_vector, magnetometer_vector, delta_time);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ahrs_update_no_magnetometer(Ahrs *self, PyObject *args) {
    PyArrayObject *gyroscope_array;
    PyArrayObject *accelerometer_array;
    float delta_time;

    const char *error = PARSE_TUPLE(args, "O!O!f", &PyArray_Type, &gyroscope_array, &PyArray_Type, &accelerometer_array, &delta_time);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector gyroscope_vector;
    FusionVector accelerometer_vector;

    error = parse_array(gyroscope_vector.array, gyroscope_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    error = parse_array(accelerometer_vector.array, accelerometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionAhrsUpdateNoMagnetometer(&self->ahrs, gyroscope_vector, accelerometer_vector, delta_time);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ahrs_update_external_heading(Ahrs *self, PyObject *args) {
    PyArrayObject *gyroscope_array;
    PyArrayObject *accelerometer_array;
    float heading;
    float delta_time;

    const char *error = PARSE_TUPLE(args, "O!O!ff", &PyArray_Type, &gyroscope_array, &PyArray_Type, &accelerometer_array, &heading, &delta_time);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionVector gyroscope_vector;
    FusionVector accelerometer_vector;

    error = parse_array(gyroscope_vector.array, gyroscope_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    error = parse_array(accelerometer_vector.array, accelerometer_array, 3);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    FusionAhrsUpdateExternalHeading(&self->ahrs, gyroscope_vector, accelerometer_vector, heading, delta_time);
    Py_INCREF(Py_None);
    return Py_None;
}

static int ahrs_set_heading(Ahrs *self, PyObject *value, void *closure) {
    const float heading = (float) PyFloat_AsDouble(value);

    if (PyErr_Occurred()) {
        return -1;
    }

    FusionAhrsSetHeading(&self->ahrs, heading);;
    return 0;
}

static PyGetSetDef ahrs_get_set[] = {
        {"settings", NULL, (setter) ahrs_set_settings,                       "", NULL},
        {"quaternion",          (getter) ahrs_get_quaternion,          NULL, "", NULL},
        {"linear_acceleration", (getter) ahrs_get_linear_acceleration, NULL, "", NULL},
        {"earth_acceleration",  (getter) ahrs_get_earth_acceleration,  NULL, "", NULL},
        {"internal_states",     (getter) ahrs_get_internal_states,     NULL, "", NULL},
        {"flags",               (getter) ahrs_get_flags,               NULL, "", NULL},
        {"heading",  NULL, (setter) ahrs_set_heading,                        "", NULL},
        {NULL}  /* sentinel */
};

static PyMethodDef ahrs_methods[] = {
        {"reset",                   (PyCFunction) ahrs_reset,                   METH_NOARGS,  ""},
        {"update",                  (PyCFunction) ahrs_update,                  METH_VARARGS, ""},
        {"update_no_magnetometer",  (PyCFunction) ahrs_update_no_magnetometer,  METH_VARARGS, ""},
        {"update_external_heading", (PyCFunction) ahrs_update_external_heading, METH_VARARGS, ""},
        {NULL} /* sentinel */
};

static PyTypeObject ahrs_object = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "imufusion.Ahrs",
        .tp_basicsize = sizeof(Ahrs),
        .tp_dealloc = (destructor) ahrs_free,
        .tp_new = ahrs_new,
        .tp_getset = ahrs_get_set,
        .tp_methods = ahrs_methods,
};

#endif
