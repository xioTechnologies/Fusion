#ifndef HELPERS_H
#define HELPERS_H

#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>
#include <string.h>

#define PARSE_TUPLE(args, format, ...) (PyArg_ParseTuple(args, format, __VA_ARGS__) == 0 ? create_parse_tuple_error_string(format) : NULL)

static void concatenate(char *const destination, const size_t destination_size, const char *const source) {
    strncat(destination, source, destination_size - strlen(destination) - 1);
}

static char *const create_parse_tuple_error_string(const char *format) {
    static char string[256] = "Arguments are not (";

    while (*format != '\0') {
        switch (*format) {
            case 'I':
                concatenate(string, sizeof(string), "unsigned int");
                break;
            case 'O':
                concatenate(string, sizeof(string), "numpy.array");
                break;
            case 'f':
                concatenate(string, sizeof(string), "float");
                break;
            case 'l':
                concatenate(string, sizeof(string), "long int");
                break;
            default:
                concatenate(string, sizeof(string), "unknown type");
                break;
        }

        do {
            format++;
        } while (*format == '!');

        if (*format != '\0') {
            concatenate(string, sizeof(string), ", ");
        } else {
            concatenate(string, sizeof(string), ")");
        }
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
