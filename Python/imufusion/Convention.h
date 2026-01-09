#ifndef CONVENTION_H
#define CONVENTION_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

static int convention_from(FusionConvention *const convention, const int convention_int) {
    switch (convention_int) {
        case FusionConventionNwu:
        case FusionConventionEnu:
        case FusionConventionNed:
            *convention = (FusionConvention) convention_int;
            return 0;
    }

    PyErr_SetString(PyExc_ValueError, "'convention' must be imufusion.CONVENTION_*");
    return -1;
}

#endif
