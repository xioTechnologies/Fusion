#ifndef HELPERS_H
#define HELPERS_H

#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>

#define PARSE_TUPLE(args, format, ...) (PyArg_ParseTuple(args, format, __VA_ARGS__) == 0 ? create_parse_tuple_error_string(format) : NULL)

static char *const create_parse_tuple_error_string(const char *format) {
    static char string[256] = "Arguments are not (";

    while (*format != '\0') {
        char new_string[sizeof(string)];

        switch (*format) {
            case 'I':
                snprintf(new_string, sizeof(new_string), "%s%s", string, "unsigned int");
                break;
            case 'O':
                snprintf(new_string, sizeof(new_string), "%s%s", string, "numpy.array");
                break;
            case 'f':
                snprintf(new_string, sizeof(new_string), "%s%s", string, "float");
                break;
            case 'l':
                snprintf(new_string, sizeof(new_string), "%s%s", string, "long int");
                break;
            default:
                snprintf(new_string, sizeof(new_string), "%s%s", string, "unknown type");
                break;
        }
        snprintf(string, sizeof(string), "%s", new_string);

        do {
            format++;
        } while (*format == '!');

        if (*format != '\0') {
            snprintf(new_string, sizeof(new_string), "%s%s", string, ", ");
        } else {
            snprintf(new_string, sizeof(new_string), "%s%s", string, ")");
        }
        snprintf(string, sizeof(string), "%s", new_string);
    }
    return string;
}

static char *const parse_array(float *const destination, const PyArrayObject *const array, const int size) {
    if (PyArray_NDIM(array) != 1) {
        return "Array dimensions is not 1";
    }

    if (PyArray_SIZE(array) != size) {
        static char string[32];
        snprintf(string, sizeof(string), "Array size is not %u", size);
        return string;
    }

    int offset = 0;

    for (int index = 0; index < size; index++) {
        PyObject *object = PyArray_GETITEM(array, PyArray_BYTES(array) + offset);

        destination[index] = (float) PyFloat_AsDouble(object);
        Py_DECREF(object);

        if (PyErr_Occurred()) {
            return "Invalid array element type";
        }

        offset += (int) PyArray_STRIDE(array, 0);
    }

    return NULL;
}

static PyObject *const build_bool(const bool value) {
    return Py_BuildValue("O", value ? Py_True : Py_False);
}

#endif
