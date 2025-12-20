#include "Ahrs.h"
#include "AhrsFlags.h"
#include "AhrsInternalStates.h"
#include "AhrsSettings.h"
#include "Bias.h"
#include "BiasSettings.h"
#include "Compass.h"
#include "Convert.h"
#include "Model.h"
#include <Python.h>
#include "Remap.h"

static struct PyModuleDef config = {
    PyModuleDef_HEAD_INIT,
    "imufusion",
    "",
    -1,
};

bool add_object(PyObject *const module, PyTypeObject *const type_object, const char *const name) {
    if (PyType_Ready(type_object) != 0) {
        return false;
    }

    Py_INCREF(type_object);

    if (PyModule_AddObject(module, name, (PyObject *) type_object) == 0) {
        return true;
    }

    Py_DECREF(type_object);
    return false;
}

PyMODINIT_FUNC PyInit_imufusion() {
    import_array();

    PyObject *const module = PyModule_Create(&config);

    if (module != NULL &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXPYPZ", FusionRemapAlignmentPXPYPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXNZPY", FusionRemapAlignmentPXNZPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXNYNZ", FusionRemapAlignmentPXNYNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PXPZNY", FusionRemapAlignmentPXPZNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXPYNZ", FusionRemapAlignmentNXPYNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXPZPY", FusionRemapAlignmentNXPZPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXNYPZ", FusionRemapAlignmentNXNYPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NXNZNY", FusionRemapAlignmentNXNZNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYNXPZ", FusionRemapAlignmentPYNXPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYNZNX", FusionRemapAlignmentPYNZNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYPXNZ", FusionRemapAlignmentPYPXNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PYPZPX", FusionRemapAlignmentPYPZPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYPXPZ", FusionRemapAlignmentNYPXPZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYNZPX", FusionRemapAlignmentNYNZPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYNXNZ", FusionRemapAlignmentNYNXNZ) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NYPZNX", FusionRemapAlignmentNYPZNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZPYNX", FusionRemapAlignmentPZPYNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZPXPY", FusionRemapAlignmentPZPXPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZNYPX", FusionRemapAlignmentPZNYPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_PZNXNY", FusionRemapAlignmentPZNXNY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZPYPX", FusionRemapAlignmentNZPYPX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZNXPY", FusionRemapAlignmentNZNXPY) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZNYNX", FusionRemapAlignmentNZNYNX) == 0) &&
        (PyModule_AddIntConstant(module, "ALIGNMENT_NZPXNY", FusionRemapAlignmentNZPXNY) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_NWU", FusionConventionNwu) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_ENU", FusionConventionEnu) == 0) &&
        (PyModule_AddIntConstant(module, "CONVENTION_NED", FusionConventionNed) == 0) &&
        (PyModule_AddFunctions(module, compass_methods) == 0) &&
        (PyModule_AddFunctions(module, convert_methods) == 0) &&
        (PyModule_AddFunctions(module, model_methods) == 0) &&
        (PyModule_AddFunctions(module, remap_methods) == 0) &&
        add_object(module, &ahrs_object, "Ahrs") &&
        add_object(module, &ahrs_flags_object, "AhrsFlags") &&
        add_object(module, &ahrs_internal_states_object, "AhrsInternalStates") &&
        add_object(module, &ahrs_settings_object, "AhrsSettings") &&
        add_object(module, &bias_object, "Bias") &&
        add_object(module, &bias_settings_object, "BiasSettings")) {
        return module;
    }

    Py_DECREF(module);
    return NULL;
}
