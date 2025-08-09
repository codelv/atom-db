/*
 * Copyright (c) 2025, CodeLV.
 *
 * Distributed under the terms of the MIT License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>


// This must match the structure in atom/src/catompointer.h
struct CAtomPointer
{
    PyObject* data;
};

// This must match the structure in atom/src/atomlist.h
struct AtomList
{
    PyListObject list;
    PyObject* validator;
    struct CAtomPointer* pointer;
};

static PyObject* atomlist_owner(PyObject* mod, PyObject* obj)
{
    static PyObject* atomlist = 0;
    if (!atomlist) {
        PyObject* module = PyImport_ImportModule("atom.catom");
        if (!module) {
            PyErr_SetString( PyExc_ImportError, "Could not import 'atom.catom'" );
            return 0;
        }

        atomlist = PyObject_GetAttrString(module, "atomlist");
        Py_DECREF(module);
        if ( !atomlist ) {
            PyErr_SetString( PyExc_ImportError, "Could not import 'atom.catom.atomlist'" );
            return 0;
        }
    }

    if (!PyObject_IsInstance(obj, atomlist)) {
        PyErr_SetString( PyExc_TypeError, "Argument must be an atomlist instance" );
        return 0;
    }
    struct AtomList* alist = (struct AtomList*) obj;
    if (!alist->pointer || !alist->pointer->data)
        Py_RETURN_NONE;
    return Py_NewRef(alist->pointer->data);
}

static PyMethodDef ext_methods[] = {
    {"atomlist_owner",  atomlist_owner, METH_O, "Retrive the atom from an atomlist instance."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef ext_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ext",
    .m_doc = "ext",
    .m_size = -1,
    ext_methods
};

PyMODINIT_FUNC
PyInit_ext(void)
{
    return PyModule_Create( &ext_module );
}

