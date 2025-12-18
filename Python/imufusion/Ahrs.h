#ifndef AHRS_H
#define AHRS_H

#include "../../Fusion/Fusion.h"
#include "Flags.h"
#include "InternalStates.h"
#include "NpArray.h"
#include <Python.h>
#include "Settings.h"
#include <stdio.h>

typedef struct {
    PyObject_HEAD
    FusionAhrs ahrs;
} Ahrs;

static PyObject *ahrs_new(PyTypeObject *subtype, PyObject *args, PyObject *keywords) {
    Ahrs *const self = (Ahrs *) subtype->tp_alloc(subtype, 0);

    if (self == NULL) {
        return NULL;
    }

    FusionAhrsInitialise(&self->ahrs);
    return (PyObject *) self;
}

static void ahrs_free(Ahrs *self) {
    Py_TYPE(self)->tp_free(self);
}

static int ahrs_set_settings(Ahrs *self, PyObject *value, void *closure) {
    if (!PyObject_TypeCheck(value, &settings_object)) {
        PyErr_Format(PyExc_TypeError, "Expected %s", settings_object.tp_name);
        return -1;
    }

    FusionAhrsSetSettings(&self->ahrs, &((Settings *) value)->settings);
    return 0;
}

static PyObject *ahrs_get_quaternion(Ahrs *self) {
    const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&self->ahrs);

    return np_array_1x4_from(quaternion.array);
}

static int ahrs_set_quaternion(Ahrs *self, PyObject *value, void *closure) {
    FusionQuaternion quaternion;

    if (np_array_1x4_to(quaternion.array, value) != 0) {
        return -1;
    }

    FusionAhrsSetQuaternion(&self->ahrs, quaternion);
    return 0;
}

static PyObject *ahrs_get_gravity(Ahrs *self) {
    const FusionVector gravity = FusionAhrsGetGravity(&self->ahrs);

    return np_array_1x3_from(gravity.array);
}

static PyObject *ahrs_get_linear_acceleration(Ahrs *self) {
    const FusionVector linear_acceleration = FusionAhrsGetLinearAcceleration(&self->ahrs);

    return np_array_1x3_from(linear_acceleration.array);
}

static PyObject *ahrs_get_earth_acceleration(Ahrs *self) {
    const FusionVector earth_acceleration = FusionAhrsGetEarthAcceleration(&self->ahrs);

    return np_array_1x3_from(earth_acceleration.array);
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
    Py_RETURN_NONE;
}

static PyObject *ahrs_update(Ahrs *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *accelerometer_object;
    PyObject *magnetometer_object;
    float delta_time;

    if (PyArg_ParseTuple(args, "OOOf", &gyroscope_object, &accelerometer_object, &magnetometer_object, &delta_time) == 0) {
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

    FusionAhrsUpdate(&self->ahrs, gyroscope, accelerometer, magnetometer, delta_time);
    Py_RETURN_NONE;
}

static PyObject *ahrs_update_no_magnetometer(Ahrs *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *accelerometer_object;
    float delta_time;

    if (PyArg_ParseTuple(args, "OOf", &gyroscope_object, &accelerometer_object, &delta_time) == 0) {
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

    FusionAhrsUpdateNoMagnetometer(&self->ahrs, gyroscope, accelerometer, delta_time);
    Py_RETURN_NONE;
}

static PyObject *ahrs_update_external_heading(Ahrs *self, PyObject *args) {
    PyObject *gyroscope_object;
    PyObject *accelerometer_object;
    float heading;
    float delta_time;

    if (PyArg_ParseTuple(args, "OOff", &gyroscope_object, &accelerometer_object, &heading, &delta_time) == 0) {
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

    FusionAhrsUpdateExternalHeading(&self->ahrs, gyroscope, accelerometer, heading, delta_time);
    Py_RETURN_NONE;
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
    {"settings", NULL, (setter) ahrs_set_settings, "", NULL},
    {"quaternion", (getter) ahrs_get_quaternion, (setter) ahrs_set_quaternion, "", NULL},
    {"gravity", (getter) ahrs_get_gravity, NULL, "", NULL},
    {"linear_acceleration", (getter) ahrs_get_linear_acceleration, NULL, "", NULL},
    {"earth_acceleration", (getter) ahrs_get_earth_acceleration, NULL, "", NULL},
    {"internal_states", (getter) ahrs_get_internal_states, NULL, "", NULL},
    {"flags", (getter) ahrs_get_flags, NULL, "", NULL},
    {"heading", NULL, (setter) ahrs_set_heading, "", NULL},
    {NULL} /* sentinel */
};

static PyMethodDef ahrs_methods[] = {
    {"reset", (PyCFunction) ahrs_reset, METH_NOARGS, ""},
    {"update", (PyCFunction) ahrs_update, METH_VARARGS, ""},
    {"update_no_magnetometer", (PyCFunction) ahrs_update_no_magnetometer, METH_VARARGS, ""},
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
