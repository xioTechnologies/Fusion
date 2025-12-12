#include "Ahrs.h"
#include "Compass.h"
#include "Flags.h"
#include "InternalStates.h"
#include <numpy/arrayobject.h>
#include "Bias.h"
#include <Python.h>
#include "Quaternion.h"
#include "Settings.h"
#include "Swap.h"

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
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXPYPZ", FusionSwapAlignmentPXPYPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXNZPY", FusionSwapAlignmentPXNZPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXNYNZ", FusionSwapAlignmentPXNYNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXPZNY", FusionSwapAlignmentPXPZNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXPYNZ", FusionSwapAlignmentNXPYNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXPZPY", FusionSwapAlignmentNXPZPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXNYPZ", FusionSwapAlignmentNXNYPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXNZNY", FusionSwapAlignmentNXNZNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYNXPZ", FusionSwapAlignmentPYNXPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYNZNX", FusionSwapAlignmentPYNZNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYPXNZ", FusionSwapAlignmentPYPXNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYPZPX", FusionSwapAlignmentPYPZPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYPXPZ", FusionSwapAlignmentNYPXPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYNZPX", FusionSwapAlignmentNYNZPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYNXNZ", FusionSwapAlignmentNYNXNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYPZNX", FusionSwapAlignmentNYPZNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZPYNX", FusionSwapAlignmentPZPYNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZPXPY", FusionSwapAlignmentPZPXPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZNYPX", FusionSwapAlignmentPZNYPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZNXNY", FusionSwapAlignmentPZNXNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZPYPX", FusionSwapAlignmentNZPYPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZNXPY", FusionSwapAlignmentNZNXPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZNYNX", FusionSwapAlignmentNZNYNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZPXNY", FusionSwapAlignmentNZPXNY) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_NWU", FusionConventionNwu) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_ENU", FusionConventionEnu) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_NED", FusionConventionNed) == 0) &&
        (PyModule_AddFunctions(module, compass_methods) == 0) &&
        (PyModule_AddFunctions(module, swap_methods) == 0) &&
        add_object(module, &ahrs_object, "Ahrs") &&
        add_object(module, &bias_object, "Offset") &&
        add_object(module, &flags_object, "Flags") &&
        add_object(module, &internal_states_object, "InternalStates") &&
        add_object(module, &settings_object, "Settings") &&
        add_object(module, &quaternion_object, "Quaternion")) {
        return module;
    }
    Py_DECREF(module);
    return NULL;
}
