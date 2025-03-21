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

static int ahrs_set_quaternion(Ahrs *self, PyObject *value, void *closure) {
    if (PyObject_IsInstance(value, (PyObject *) &quaternion_object) == false) {
        static char string[64];
        snprintf(string, sizeof(string), "Value type is not %s", quaternion_object.tp_name);
        PyErr_SetString(PyExc_TypeError, string);
        return -1;
    }
    FusionAhrsSetQuaternion(&self->ahrs, ((Quaternion *) value)->quaternion);
    return 0;
}

static PyObject *ahrs_get_gravity(Ahrs *self) {
    FusionVector *const gravity = malloc(sizeof(FusionVector));
    *gravity = FusionAhrsGetGravity(&self->ahrs);

    const npy_intp dims[] = {3};
    PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, gravity->array);
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
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

static PyObject *ahrs_update_batch(Ahrs *self, PyObject *args) {
    PyArrayObject *gyroscope_array;
    PyArrayObject *accelerometer_array;
    PyArrayObject *magnetometer_array;
    PyArrayObject *delta_time_array;
    
    const char *error = PARSE_TUPLE(args, "O!O!O!O!", 
                                    &PyArray_Type, &gyroscope_array, 
                                    &PyArray_Type, &accelerometer_array, 
                                    &PyArray_Type, &magnetometer_array,
                                    &PyArray_Type, &delta_time_array);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }

    if (PyArray_TYPE(gyroscope_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "gyroscope array must be np.float32");
        return NULL;
    }
    if (PyArray_TYPE(accelerometer_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "accelerometer array must be np.float32");
        return NULL;
    }
    if (PyArray_TYPE(magnetometer_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "magnetometer array must be np.float32");
        return NULL;
    }
    if (PyArray_TYPE(delta_time_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "delta_time array must be np.float32");
        return NULL;
    }
    
    npy_intp n = PyArray_SIZE(gyroscope_array) / 3;
    if ((PyArray_SIZE(accelerometer_array) / 3) != n ||
        (PyArray_SIZE(magnetometer_array) / 3) != n ||
        (PyArray_SIZE(delta_time_array)) != n) {
        PyErr_SetString(PyExc_ValueError, "모든 입력 배열의 크기가 일치해야 합니다.");
        return NULL;
    }
    
    npy_intp dims[2] = { n, 3 };
    PyObject *euler_array = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float *euler_data = (float *) PyArray_DATA((PyArrayObject *)euler_array);
    PyObject *earth_accel_array = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float *earth_accel_data = (float *) PyArray_DATA((PyArrayObject *)earth_accel_array);
    
    float *gyro_data = (float *) PyArray_DATA(gyroscope_array);
    float *accel_data = (float *) PyArray_DATA(accelerometer_array);
    float *mag_data = (float *) PyArray_DATA(magnetometer_array);
    float *dt_data = (float *) PyArray_DATA(delta_time_array);
    
    FusionVector gyro_vec, accel_vec, mag_vec, earth_accel;
    FusionEuler euler;

    for (npy_intp i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            gyro_vec.array[j] = gyro_data[i * 3 + j];
            accel_vec.array[j] = accel_data[i * 3 + j];
            mag_vec.array[j]   = mag_data[i * 3 + j];
        }
        
        float dt = dt_data[i];
        
        FusionAhrsUpdate(&self->ahrs, gyro_vec, accel_vec, mag_vec, dt);
        
        euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&self->ahrs));
        earth_accel = FusionAhrsGetEarthAcceleration(&self->ahrs);

        euler_data[i * 3 + 0] = euler.angle.roll;
        euler_data[i * 3 + 1] = euler.angle.pitch;
        euler_data[i * 3 + 2] = euler.angle.yaw;

        earth_accel_data[i * 3 + 0] = earth_accel.axis.x;
        earth_accel_data[i * 3 + 1] = earth_accel.axis.y;
        earth_accel_data[i * 3 + 2] = earth_accel.axis.z;
    }

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, euler_array);
    PyTuple_SET_ITEM(result, 1, earth_accel_array);

    return result;
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


static PyObject *ahrs_update_no_magnetometer_batch(Ahrs *self, PyObject *args) {
    PyArrayObject *gyroscope_array;
    PyArrayObject *accelerometer_array;
    PyArrayObject *delta_time_array;
    
    const char *error = PARSE_TUPLE(args, "O!O!O!", 
                                    &PyArray_Type, &gyroscope_array, 
                                    &PyArray_Type, &accelerometer_array, 
                                    &PyArray_Type, &delta_time_array);
    if (error != NULL) {
        PyErr_SetString(PyExc_TypeError, error);
        return NULL;
    }
    if (PyArray_TYPE(gyroscope_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "gyroscope array must be np.float32");
        return NULL;
    }
    if (PyArray_TYPE(accelerometer_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "accelerometer array must be np.float32");
        return NULL;
    }
    if (PyArray_TYPE(delta_time_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "delta_time array must be np.float32");
        return NULL;
    }
    
    npy_intp n = PyArray_SIZE(gyroscope_array) / 3;
    if ((PyArray_SIZE(accelerometer_array) / 3) != n ||
        (PyArray_SIZE(delta_time_array)) != n) {
        PyErr_SetString(PyExc_ValueError, "모든 입력 배열의 크기가 일치해야 합니다.");
        return NULL;
    }
    
    npy_intp dims[2] = { n, 3 };
    PyObject *euler_array = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float *euler_data = (float *) PyArray_DATA((PyArrayObject *)euler_array);
    PyObject *earth_accel_array = PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float *earth_accel_data = (float *) PyArray_DATA((PyArrayObject *)earth_accel_array);
    
    float *gyro_data = (float *) PyArray_DATA(gyroscope_array);
    float *accel_data = (float *) PyArray_DATA(accelerometer_array);
    float *dt_data = (float *) PyArray_DATA(delta_time_array);
    
    FusionVector gyro_vec, accel_vec, earth_accel;
    FusionEuler euler;
    
    for (npy_intp i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            gyro_vec.array[j] = gyro_data[i * 3 + j];
            accel_vec.array[j] = accel_data[i * 3 + j];
        }
        
        float dt = dt_data[i];
        
        FusionAhrsUpdateNoMagnetometer(&self->ahrs, gyro_vec, accel_vec, dt);
        
        euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&self->ahrs));
        earth_accel = FusionAhrsGetEarthAcceleration(&self->ahrs);
        
        euler_data[i * 3 + 0] = euler.angle.roll;
        euler_data[i * 3 + 1] = euler.angle.pitch;
        euler_data[i * 3 + 2] = euler.angle.yaw;

        earth_accel_data[i * 3 + 0] = earth_accel.axis.x;
        earth_accel_data[i * 3 + 1] = earth_accel.axis.y;
        earth_accel_data[i * 3 + 2] = earth_accel.axis.z;
    }

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, euler_array);
    PyTuple_SET_ITEM(result, 1, earth_accel_array);

    return result;
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
        {"settings", NULL,                                    (setter) ahrs_set_settings,   "", NULL},
        {"quaternion",          (getter) ahrs_get_quaternion, (setter) ahrs_set_quaternion, "", NULL},
        {"gravity",             (getter) ahrs_get_gravity,             NULL,                "", NULL},
        {"linear_acceleration", (getter) ahrs_get_linear_acceleration, NULL,                "", NULL},
        {"earth_acceleration",  (getter) ahrs_get_earth_acceleration,  NULL,                "", NULL},
        {"internal_states",     (getter) ahrs_get_internal_states,     NULL,                "", NULL},
        {"flags",               (getter) ahrs_get_flags,               NULL,                "", NULL},
        {"heading",  NULL,                                    (setter) ahrs_set_heading,    "", NULL},
        {NULL}  /* sentinel */
};

static PyMethodDef ahrs_methods[] = {
        {"reset",                        (PyCFunction) ahrs_reset,                        METH_NOARGS,  ""},
        {"update",                       (PyCFunction) ahrs_update,                       METH_VARARGS, ""},
        {"update_no_magnetometer",       (PyCFunction) ahrs_update_no_magnetometer,       METH_VARARGS, ""},
        {"update_external_heading",      (PyCFunction) ahrs_update_external_heading,      METH_VARARGS, ""},
        {"update_batch",                 (PyCFunction) ahrs_update_batch,                 METH_VARARGS, ""},
        {"update_no_magnetometer_batch", (PyCFunction) ahrs_update_no_magnetometer_batch, METH_VARARGS, ""},
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
