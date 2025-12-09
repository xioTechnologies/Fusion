#include "Ahrs.h"
#include "Axes.h"
#include "Compass.h"
#include "Flags.h"
#include "InternalStates.h"
#include <numpy/arrayobject.h>
#include "Offset.h"
#include <Python.h>
#include "Quaternion.h"
#include "Settings.h"

static struct PyModuleDef config = {
        PyModuleDef_HEAD_INIT,
        "imufusion",
        "",
        -1,
};

bool add_object(PyObject *const module, PyTypeObject *const type_object, const char *const name) {
    if (PyType_Ready(type_object) == 0) {
        Py_INCREF(type_object);
        if (PyModule_AddObject(module, name, (PyObject *) type_object) == 0) {
            return true;
        }
        Py_DECREF(type_object);
        return false;
    }
    return false;
}

PyMODINIT_FUNC PyInit_imufusion() {
    import_array();

    PyObject *const module = PyModule_Create(&config);

    if (module != NULL &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXPYPZ", FusionAxesAlignmentPXPYPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXNZPY", FusionAxesAlignmentPXNZPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXNYNZ", FusionAxesAlignmentPXNYNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXPZNY", FusionAxesAlignmentPXPZNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXPYNZ", FusionAxesAlignmentNXPYNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXPZPY", FusionAxesAlignmentNXPZPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXNYPZ", FusionAxesAlignmentNXNYPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXNZNY", FusionAxesAlignmentNXNZNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYNXPZ", FusionAxesAlignmentPYNXPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYNZNX", FusionAxesAlignmentPYNZNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYPXNZ", FusionAxesAlignmentPYPXNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYPZPX", FusionAxesAlignmentPYPZPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYPXPZ", FusionAxesAlignmentNYPXPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYNZPX", FusionAxesAlignmentNYNZPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYNXNZ", FusionAxesAlignmentNYNXNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYPZNX", FusionAxesAlignmentNYPZNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZPYNX", FusionAxesAlignmentPZPYNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZPXPY", FusionAxesAlignmentPZPXPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZNYPX", FusionAxesAlignmentPZNYPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZNXNY", FusionAxesAlignmentPZNXNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZPYPX", FusionAxesAlignmentNZPYPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZNXPY", FusionAxesAlignmentNZNXPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZNYNX", FusionAxesAlignmentNZNYNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZPXNY", FusionAxesAlignmentNZPXNY) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_NWU", FusionConventionNwu) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_ENU", FusionConventionEnu) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_NED", FusionConventionNed) == 0) &&
        (PyModule_AddFunctions(module, axes_methods) == 0) &&
        (PyModule_AddFunctions(module, compass_methods) == 0) &&
        add_object(module, &ahrs_object, "Ahrs") &&
        add_object(module, &flags_object, "Flags") &&
        add_object(module, &internal_states_object, "InternalStates") &&
        add_object(module, &offset_object, "Offset") &&
        add_object(module, &settings_object, "Settings") &&
        add_object(module, &quaternion_object, "Quaternion")) {
        return module;
    }
    Py_DECREF(module);
    return NULL;
}
