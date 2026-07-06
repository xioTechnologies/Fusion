#ifndef AHRS_H
#define AHRS_H

#include "../../Fusion/Fusion.h"
#include "AhrsFlags.h"
#include "AhrsInternalStates.h"
#include "AhrsSettings.h"
#include "NpArray.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD
    FusionAhrs wrapped;
} Ahrs;

static PyObject *ahrs_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    if (PyArg_ParseTuple(args, "") == 0) {
        return NULL;
    }

    Ahrs *const self = (Ahrs *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    FusionAhrsInitialise(&self->wrapped);
    return (PyObject *) self;
}

static void ahrs_free(Ahrs *self) {
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ahrs_restart(Ahrs *self, PyObject *args) {
    FusionAhrsRestart(&self->wrapped);
    Py_RETURN_NONE;
}

static PyObject *ahrs_set_settings(Ahrs *self, PyObject *arg) {
    AhrsSettings *settings;

    if (PyArg_Parse(arg, "O!", &ahrs_settings_object, &settings) == 0) {
        return NULL;
    }

    FusionAhrsSetSettings(&self->wrapped, &settings->wrapped);
    Py_RETURN_NONE;
}

static PyObject *ahrs_set_sample_period(Ahrs *self, PyObject *arg) {
    const float sample_period = (float) PyFloat_AsDouble(arg);

    if (PyErr_Occurred()) {
        return NULL;
    }

    FusionAhrsSetSamplePeriod(&self->wrapped, sample_period);
    Py_RETURN_NONE;
}

static PyObject *ahrs_update(Ahrs *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *accelerometer_object;
    PyObject *magnetometer_object;

    if (PyArg_ParseTuple(args, "OOO", &gyroscope_object, &accelerometer_object, &magnetometer_object) == 0) {
        return NULL;
    }

    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, gyroscope_object) != 0) {
        return NULL;
    }

    FusionVector accelerometer;

    if (np_array_1x3_to(accelerometer.array, accelerometer_object) != 0) {
        return NULL;
    }

    FusionVector magnetometer;

    if (np_array_1x3_to(magnetometer.array, magnetometer_object) != 0) {
        return NULL;
    }

    FusionAhrsUpdate(&self->wrapped, gyroscope, accelerometer, magnetometer);
    Py_RETURN_NONE;
}

static PyObject *ahrs_update_no_magnetometer(Ahrs *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *accelerometer_object;

    if (PyArg_ParseTuple(args, "OO", &gyroscope_object, &accelerometer_object) == 0) {
        return NULL;
    }

    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, gyroscope_object) != 0) {
        return NULL;
    }

    FusionVector accelerometer;

    if (np_array_1x3_to(accelerometer.array, accelerometer_object) != 0) {
        return NULL;
    }

    FusionAhrsUpdateNoMagnetometer(&self->wrapped, gyroscope, accelerometer);
    Py_RETURN_NONE;
}

static PyObject *ahrs_update_external_heading(Ahrs *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *accelerometer_object;
    float heading;

    if (PyArg_ParseTuple(args, "OOf", &gyroscope_object, &accelerometer_object, &heading) == 0) {
        return NULL;
    }

    FusionVector gyroscope;

    if (np_array_1x3_to(gyroscope.array, gyroscope_object) != 0) {
        return NULL;
    }

    FusionVector accelerometer;

    if (np_array_1x3_to(accelerometer.array, accelerometer_object) != 0) {
        return NULL;
    }

    FusionAhrsUpdateExternalHeading(&self->wrapped, gyroscope, accelerometer, heading);
    Py_RETURN_NONE;
}

static PyObject *ahrs_get_quaternion(Ahrs *self, PyObject *args) {
    const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&self->wrapped);

    return np_array_1x4_from(quaternion.array);
}

static PyObject *ahrs_set_quaternion(Ahrs *self, PyObject *arg) {
    FusionQuaternion quaternion;

    if (np_array_1x4_to(quaternion.array, arg) != 0) {
        return NULL;
    }

    FusionAhrsSetQuaternion(&self->wrapped, quaternion);
    Py_RETURN_NONE;
}

static PyObject *ahrs_get_gravity(Ahrs *self, PyObject *args) {
    const FusionVector gravity = FusionAhrsGetGravity(&self->wrapped);

    return np_array_1x3_from(gravity.array);
}

static PyObject *ahrs_get_linear_acceleration(Ahrs *self, PyObject *args) {
    const FusionVector linear_acceleration = FusionAhrsGetLinearAcceleration(&self->wrapped);

    return np_array_1x3_from(linear_acceleration.array);
}

static PyObject *ahrs_get_earth_acceleration(Ahrs *self, PyObject *args) {
    const FusionVector earth_acceleration = FusionAhrsGetEarthAcceleration(&self->wrapped);

    return np_array_1x3_from(earth_acceleration.array);
}

static PyObject *ahrs_get_internal_states(Ahrs *self, PyObject *args) {
    const FusionAhrsInternalStates internal_states = FusionAhrsGetInternalStates(&self->wrapped);

    return internal_states_from(&internal_states);
}

static PyObject *ahrs_get_flags(Ahrs *self, PyObject *args) {
    const FusionAhrsFlags flags = FusionAhrsGetFlags(&self->wrapped);

    return ahrs_flags_from(&flags);
}

static PyObject *ahrs_set_heading(Ahrs *self, PyObject *arg) {
    const float heading = (float) PyFloat_AsDouble(arg);

    if (PyErr_Occurred()) {
        return NULL;
    }

    FusionAhrsSetHeading(&self->wrapped, heading);
    Py_RETURN_NONE;
}

static PyMethodDef ahrs_methods[] = {
    {"restart", (PyCFunction) ahrs_restart, METH_NOARGS, ""},
    {"set_settings", (PyCFunction) ahrs_set_settings, METH_O, ""},
    {"set_sample_period", (PyCFunction) ahrs_set_sample_period, METH_O, ""},
    {"update", (PyCFunction) ahrs_update, METH_VARARGS, ""},
    {"update_no_magnetometer", (PyCFunction) ahrs_update_no_magnetometer, METH_VARARGS, ""},
    {"update_external_heading", (PyCFunction) ahrs_update_external_heading, METH_VARARGS, ""},
    {"get_quaternion", (PyCFunction) ahrs_get_quaternion, METH_NOARGS, ""},
    {"set_quaternion", (PyCFunction) ahrs_set_quaternion, METH_O, ""},
    {"get_gravity", (PyCFunction) ahrs_get_gravity, METH_NOARGS, ""},
    {"get_linear_acceleration", (PyCFunction) ahrs_get_linear_acceleration, METH_NOARGS, ""},
    {"get_earth_acceleration", (PyCFunction) ahrs_get_earth_acceleration, METH_NOARGS, ""},
    {"get_internal_states", (PyCFunction) ahrs_get_internal_states, METH_NOARGS, ""},
    {"get_flags", (PyCFunction) ahrs_get_flags, METH_NOARGS, ""},
    {"set_heading", (PyCFunction) ahrs_set_heading, METH_O, ""},
    {NULL} /* sentinel */
};

static PyTypeObject ahrs_object = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "imufusion.Ahrs",
    .tp_basicsize = sizeof(Ahrs),
    .tp_dealloc = (destructor) ahrs_free,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ahrs_new,
    .tp_methods = ahrs_methods,
};

#endif
