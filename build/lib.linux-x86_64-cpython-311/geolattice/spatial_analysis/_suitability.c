/*
 * GeoLattice suitability analysis — NumPy C extension.
 *
 * Implements ArcGIS Spatial Analyst-style array kernels:
 *   reclassify, rescale_by_function, weighted_overlay, weighted_sum,
 *   fuzzy_membership, fuzzy_overlay, locate_regions
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    MEMBER_GAUSSIAN = 0,
    MEMBER_NEAR = 1,
    MEMBER_LARGE = 2,
    MEMBER_SMALL = 3,
    MEMBER_MSLARGE = 4,
    MEMBER_MSSMALL = 5,
    MEMBER_LINEAR = 6
};

enum {
    OVERLAY_AND = 0,
    OVERLAY_OR = 1,
    OVERLAY_PRODUCT = 2,
    OVERLAY_SUM = 3,
    OVERLAY_GAMMA = 4
};

enum {
    RESCALE_GAUSSIAN = 0,
    RESCALE_NEAR = 1,
    RESCALE_LARGE = 2,
    RESCALE_SMALL = 3,
    RESCALE_MSLARGE = 4,
    RESCALE_MSSMALL = 5,
    RESCALE_LINEAR = 6,
    RESCALE_EXPONENTIAL = 7,
    RESCALE_LOGARITHM = 8,
    RESCALE_POWER = 9,
    RESCALE_LOGISTIC_GROWTH = 10,
    RESCALE_LOGISTIC_DECAY = 11,
    RESCALE_SYMMETRIC_LINEAR = 12
};

static int
parse_membership_type(const char *name)
{
    if (!name) {
        return -1;
    }
    if (strcmp(name, "gaussian") == 0) {
        return MEMBER_GAUSSIAN;
    }
    if (strcmp(name, "near") == 0) {
        return MEMBER_NEAR;
    }
    if (strcmp(name, "large") == 0) {
        return MEMBER_LARGE;
    }
    if (strcmp(name, "small") == 0) {
        return MEMBER_SMALL;
    }
    if (strcmp(name, "mslarge") == 0 || strcmp(name, "ms_large") == 0) {
        return MEMBER_MSLARGE;
    }
    if (strcmp(name, "mssmall") == 0 || strcmp(name, "ms_small") == 0) {
        return MEMBER_MSSMALL;
    }
    if (strcmp(name, "linear") == 0) {
        return MEMBER_LINEAR;
    }
    return -1;
}

static int
parse_overlay_type(const char *name)
{
    if (!name) {
        return -1;
    }
    if (strcmp(name, "and") == 0) {
        return OVERLAY_AND;
    }
    if (strcmp(name, "or") == 0) {
        return OVERLAY_OR;
    }
    if (strcmp(name, "product") == 0) {
        return OVERLAY_PRODUCT;
    }
    if (strcmp(name, "sum") == 0) {
        return OVERLAY_SUM;
    }
    if (strcmp(name, "gamma") == 0) {
        return OVERLAY_GAMMA;
    }
    return -1;
}

static int
parse_rescale_type(const char *name)
{
    int m;
    if (!name) {
        return -1;
    }
    m = parse_membership_type(name);
    if (m >= 0) {
        return m;
    }
    if (strcmp(name, "exponential") == 0) {
        return RESCALE_EXPONENTIAL;
    }
    if (strcmp(name, "logarithm") == 0 || strcmp(name, "log") == 0) {
        return RESCALE_LOGARITHM;
    }
    if (strcmp(name, "power") == 0) {
        return RESCALE_POWER;
    }
    if (strcmp(name, "logistic_growth") == 0 || strcmp(name, "logisticgrowth") == 0) {
        return RESCALE_LOGISTIC_GROWTH;
    }
    if (strcmp(name, "logistic_decay") == 0 || strcmp(name, "logisticdecay") == 0) {
        return RESCALE_LOGISTIC_DECAY;
    }
    if (strcmp(name, "symmetric_linear") == 0 || strcmp(name, "symmetriclinear") == 0) {
        return RESCALE_SYMMETRIC_LINEAR;
    }
    return -1;
}

static double
clamp01(double v)
{
    if (v < 0.0) {
        return 0.0;
    }
    if (v > 1.0) {
        return 1.0;
    }
    return v;
}

static double
fuzzy_value(int kind, double x, double p0, double p1, double mean, double std)
{
    double mid, spread, a, b, denom, u, lo, hi;

    switch (kind) {
    case MEMBER_GAUSSIAN:
        /* μ = exp(-spread * (x - midpoint)^2) */
        mid = p0;
        spread = p1;
        return clamp01(exp(-spread * (x - mid) * (x - mid)));

    case MEMBER_NEAR:
        /* μ = 1 / (1 + spread * (x - midpoint)^2) */
        mid = p0;
        spread = p1;
        return clamp01(1.0 / (1.0 + spread * (x - mid) * (x - mid)));

    case MEMBER_LARGE:
        /* μ = 1 / (1 + (x / midpoint)^(-spread)); midpoint -> 0.5 */
        mid = p0;
        spread = p1;
        if (mid == 0.0) {
            return 0.0;
        }
        if (x == 0.0) {
            return 0.0;
        }
        return clamp01(1.0 / (1.0 + pow(x / mid, -spread)));

    case MEMBER_SMALL:
        /* μ = 1 / (1 + (x / midpoint)^(spread)); midpoint -> 0.5 */
        mid = p0;
        spread = p1;
        if (mid == 0.0) {
            return (x == 0.0) ? 1.0 : 0.0;
        }
        if (x == 0.0) {
            return 1.0;
        }
        return clamp01(1.0 / (1.0 + pow(x / mid, spread)));

    case MEMBER_MSLARGE:
        /* p0 = mean multiplier a, p1 = std multiplier b */
        a = p0;
        b = p1;
        if (std <= 0.0) {
            return (x > a * mean) ? 1.0 : 0.0;
        }
        if (x > a * mean) {
            denom = x - a * mean + b * std;
            if (denom == 0.0) {
                return 0.0;
            }
            u = 1.0 - (b * std) / denom;
            return clamp01(u);
        }
        return 0.0;

    case MEMBER_MSSMALL:
        a = p0;
        b = p1;
        if (std <= 0.0) {
            return (x > a * mean) ? 0.0 : 1.0;
        }
        if (x > a * mean) {
            denom = x - a * mean + b * std;
            if (denom == 0.0) {
                return 1.0;
            }
            u = (b * std) / denom;
            return clamp01(u);
        }
        return 1.0;

    case MEMBER_LINEAR:
        lo = p0;
        hi = p1;
        if (lo == hi) {
            return (x >= hi) ? 1.0 : 0.0;
        }
        if (lo < hi) {
            if (x <= lo) {
                return 0.0;
            }
            if (x >= hi) {
                return 1.0;
            }
            return (x - lo) / (hi - lo);
        }
        /* reverse linear */
        if (x <= hi) {
            return 1.0;
        }
        if (x >= lo) {
            return 0.0;
        }
        return (lo - x) / (lo - hi);

    default:
        return 0.0;
    }
}

static double
rescale_unit(int kind, double x, double p0, double p1, double p2, double mean, double std)
{
    double u, mid, lo, hi, shift, factor, ymin;

    switch (kind) {
    case RESCALE_GAUSSIAN:
    case RESCALE_NEAR:
    case RESCALE_LARGE:
    case RESCALE_SMALL:
    case RESCALE_MSLARGE:
    case RESCALE_MSSMALL:
    case RESCALE_LINEAR:
        return fuzzy_value(kind, x, p0, p1, mean, std);

    case RESCALE_EXPONENTIAL:
        /* preference grows exponentially toward large values */
        shift = p0;
        factor = (p1 == 0.0) ? 1.0 : p1;
        u = exp((x - shift) / factor) - 1.0;
        return u;

    case RESCALE_LOGARITHM:
        shift = p0;
        factor = (p1 == 0.0) ? 1.0 : p1;
        if (x - shift + 1.0 <= 0.0) {
            return 0.0;
        }
        return log((x - shift) * factor + 1.0);

    case RESCALE_POWER:
        shift = p0;
        factor = (p1 == 0.0) ? 1.0 : p1;
        if (x - shift < 0.0 && factor != floor(factor)) {
            return 0.0;
        }
        return pow(x - shift, factor);

    case RESCALE_LOGISTIC_GROWTH:
        lo = p0;
        hi = p1;
        ymin = p2; /* y-intercept percent 0-100, default ~1 */
        if (hi == lo) {
            return 1.0;
        }
        {
            double C = (ymin <= 0.0 || ymin >= 100.0) ? 0.01 : ymin / 100.0;
            double k = log((1.0 - C) / C) * 2.0 / (hi - lo);
            double x0 = 0.5 * (lo + hi);
            return 1.0 / (1.0 + exp(-k * (x - x0)));
        }

    case RESCALE_LOGISTIC_DECAY:
        lo = p0;
        hi = p1;
        ymin = p2;
        if (hi == lo) {
            return 0.0;
        }
        {
            double C = (ymin <= 0.0 || ymin >= 100.0) ? 0.99 : ymin / 100.0;
            double k = log(C / (1.0 - C)) * 2.0 / (hi - lo);
            double x0 = 0.5 * (lo + hi);
            return 1.0 / (1.0 + exp(k * (x - x0)));
        }

    case RESCALE_SYMMETRIC_LINEAR:
        mid = p0;
        lo = p1;
        hi = p2;
        if (x <= lo || x >= hi) {
            return 0.0;
        }
        if (x == mid) {
            return 1.0;
        }
        if (x < mid) {
            if (mid == lo) {
                return 1.0;
            }
            return (x - lo) / (mid - lo);
        }
        if (hi == mid) {
            return 1.0;
        }
        return (hi - x) / (hi - mid);

    default:
        return 0.0;
    }
}

static int
is_nodata(double v, double nodata, int has_nodata)
{
    if (isnan(v) || isinf(v)) {
        return 1;
    }
    if (has_nodata && v == nodata) {
        return 1;
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* reclassify                                                                 */
/* -------------------------------------------------------------------------- */

/*
 * reclassify(array, rules, default=nan, nodata=None)
 * rules: Nx3 float64 array of (lo, hi, new) with hi exclusive, or
 *        Nx2 encoded as lo==hi meaning exact match (lo, lo, new) via
 *        a parallel convention: if rule has shape (n, 3).
 */
static PyObject *
suit_reclassify(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"array", "rules", "default", "nodata", NULL};
    PyObject *array_obj = NULL;
    PyObject *rules_obj = NULL;
    double default_value = NAN;
    PyObject *nodata_obj = Py_None;
    PyArrayObject *arr = NULL;
    PyArrayObject *rules = NULL;
    PyArrayObject *out = NULL;
    npy_intp n, i, r, nrules;
    double nodata = 0.0;
    int has_nodata = 0;
    int nd_rules;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO|dO", kwlist, &array_obj, &rules_obj, &default_value, &nodata_obj)) {
        return NULL;
    }

    arr = (PyArrayObject *)PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    rules = (PyArrayObject *)PyArray_FROM_OTF(rules_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr || !rules) {
        goto fail;
    }
    nd_rules = PyArray_NDIM(rules);
    if (nd_rules != 2 || PyArray_DIM(rules, 1) < 2 || PyArray_DIM(rules, 1) > 3) {
        PyErr_SetString(PyExc_ValueError, "rules must be an (n, 2) or (n, 3) float array");
        goto fail;
    }
    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
        has_nodata = 1;
    }

    out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(arr), PyArray_DIMS(arr), NPY_DOUBLE);
    if (!out) {
        goto fail;
    }

    n = PyArray_SIZE(arr);
    nrules = PyArray_DIM(rules, 0);
    for (i = 0; i < n; ++i) {
        double v = ((double *)PyArray_DATA(arr))[i];
        double result = default_value;
        int matched = 0;

        if (is_nodata(v, nodata, has_nodata)) {
            ((double *)PyArray_DATA(out))[i] = has_nodata ? nodata : NAN;
            continue;
        }
        for (r = 0; r < nrules; ++r) {
            double a = *((double *)PyArray_GETPTR2(rules, r, 0));
            double b = *((double *)PyArray_GETPTR2(rules, r, 1));
            double newv;
            if (PyArray_DIM(rules, 1) == 2) {
                /* exact match: (value, new) */
                newv = b;
                if (v == a) {
                    result = newv;
                    matched = 1;
                    break;
                }
            } else {
                newv = *((double *)PyArray_GETPTR2(rules, r, 2));
                if (a == b) {
                    if (v == a) {
                        result = newv;
                        matched = 1;
                        break;
                    }
                } else if (v >= a && v < b) {
                    result = newv;
                    matched = 1;
                    break;
                }
            }
        }
        (void)matched;
        ((double *)PyArray_DATA(out))[i] = result;
    }

    Py_DECREF(arr);
    Py_DECREF(rules);
    return (PyObject *)out;

fail:
    Py_XDECREF(arr);
    Py_XDECREF(rules);
    Py_XDECREF(out);
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* fuzzy_membership                                                           */
/* -------------------------------------------------------------------------- */

static PyObject *
suit_fuzzy_membership(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "array", "membership_type", "midpoint", "spread", "minimum", "maximum",
        "mean_multiplier", "std_multiplier", "nodata", NULL};
    PyObject *array_obj = NULL;
    const char *membership_type = "gaussian";
    double midpoint = NAN;
    double spread = NAN;
    double minimum = NAN;
    double maximum = NAN;
    double mean_mult = 1.0;
    double std_mult = 1.0;
    PyObject *nodata_obj = Py_None;
    PyArrayObject *arr = NULL;
    PyArrayObject *out = NULL;
    npy_intp n, i;
    int kind;
    double nodata = 0.0;
    int has_nodata = 0;
    double mean = 0.0, std = 0.0, vmin = 0.0, vmax = 0.0;
    npy_intp count = 0;
    double p0 = 0.0, p1 = 0.0;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|sddddddO",
            kwlist,
            &array_obj,
            &membership_type,
            &midpoint,
            &spread,
            &minimum,
            &maximum,
            &mean_mult,
            &std_mult,
            &nodata_obj)) {
        return NULL;
    }

    kind = parse_membership_type(membership_type);
    if (kind < 0) {
        PyErr_Format(PyExc_ValueError, "Unknown membership_type %s", membership_type);
        return NULL;
    }

    arr = (PyArrayObject *)PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) {
        return NULL;
    }
    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            Py_DECREF(arr);
            return NULL;
        }
        has_nodata = 1;
    }

    n = PyArray_SIZE(arr);
    /* stats over valid cells */
    {
        double sum = 0.0, sumsq = 0.0;
        int first = 1;
        for (i = 0; i < n; ++i) {
            double v = ((double *)PyArray_DATA(arr))[i];
            if (is_nodata(v, nodata, has_nodata)) {
                continue;
            }
            if (first) {
                vmin = vmax = v;
                first = 0;
            } else {
                if (v < vmin) {
                    vmin = v;
                }
                if (v > vmax) {
                    vmax = v;
                }
            }
            sum += v;
            sumsq += v * v;
            count++;
        }
        if (count > 0) {
            mean = sum / (double)count;
            if (count > 1) {
                std = sqrt((sumsq - sum * sum / (double)count) / (double)(count - 1));
            } else {
                std = 0.0;
            }
        }
    }

    /* defaults */
    if (isnan(midpoint)) {
        midpoint = 0.5 * (vmin + vmax);
    }
    if (isnan(spread)) {
        if (kind == MEMBER_GAUSSIAN) {
            spread = 0.1;
        } else if (kind == MEMBER_NEAR) {
            spread = 0.1;
        } else {
            spread = 5.0;
        }
    }
    if (isnan(minimum)) {
        minimum = vmin;
    }
    if (isnan(maximum)) {
        maximum = vmax;
    }

    if (kind == MEMBER_GAUSSIAN || kind == MEMBER_NEAR || kind == MEMBER_LARGE || kind == MEMBER_SMALL) {
        p0 = midpoint;
        p1 = spread;
    } else if (kind == MEMBER_MSLARGE || kind == MEMBER_MSSMALL) {
        p0 = mean_mult;
        p1 = std_mult;
    } else {
        p0 = minimum;
        p1 = maximum;
    }

    out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(arr), PyArray_DIMS(arr), NPY_DOUBLE);
    if (!out) {
        Py_DECREF(arr);
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        double v = ((double *)PyArray_DATA(arr))[i];
        if (is_nodata(v, nodata, has_nodata)) {
            ((double *)PyArray_DATA(out))[i] = has_nodata ? nodata : NAN;
        } else {
            ((double *)PyArray_DATA(out))[i] = fuzzy_value(kind, v, p0, p1, mean, std);
        }
    }

    Py_DECREF(arr);
    return (PyObject *)out;
}

/* -------------------------------------------------------------------------- */
/* rescale_by_function                                                        */
/* -------------------------------------------------------------------------- */

static PyObject *
suit_rescale_by_function(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "array",
        "function",
        "from_scale",
        "to_scale",
        "midpoint",
        "spread",
        "minimum",
        "maximum",
        "mean_multiplier",
        "std_multiplier",
        "shift",
        "factor",
        "y_intercept_percent",
        "lower_threshold",
        "upper_threshold",
        "nodata",
        NULL};
    PyObject *array_obj = NULL;
    const char *function = "linear";
    double from_scale = 1.0;
    double to_scale = 10.0;
    double midpoint = NAN;
    double spread = NAN;
    double minimum = NAN;
    double maximum = NAN;
    double mean_mult = 1.0;
    double std_mult = 1.0;
    double shift = NAN;
    double factor = NAN;
    double y_intercept = 99.0;
    double lower_threshold = NAN;
    double upper_threshold = NAN;
    PyObject *nodata_obj = Py_None;
    PyArrayObject *arr = NULL;
    PyArrayObject *out = NULL;
    npy_intp n, i, count = 0;
    int kind;
    double nodata = 0.0;
    int has_nodata = 0;
    double mean = 0.0, std = 0.0, vmin = 0.0, vmax = 0.0;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0;
    double umin = DBL_MAX, umax = -DBL_MAX;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|sdddddddddddddO",
            kwlist,
            &array_obj,
            &function,
            &from_scale,
            &to_scale,
            &midpoint,
            &spread,
            &minimum,
            &maximum,
            &mean_mult,
            &std_mult,
            &shift,
            &factor,
            &y_intercept,
            &lower_threshold,
            &upper_threshold,
            &nodata_obj)) {
        return NULL;
    }

    kind = parse_rescale_type(function);
    if (kind < 0) {
        PyErr_Format(PyExc_ValueError, "Unknown rescale function %s", function);
        return NULL;
    }

    arr = (PyArrayObject *)PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) {
        return NULL;
    }
    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            Py_DECREF(arr);
            return NULL;
        }
        has_nodata = 1;
    }

    n = PyArray_SIZE(arr);
    {
        double sum = 0.0, sumsq = 0.0;
        int first = 1;
        for (i = 0; i < n; ++i) {
            double v = ((double *)PyArray_DATA(arr))[i];
            if (is_nodata(v, nodata, has_nodata)) {
                continue;
            }
            if (first) {
                vmin = vmax = v;
                first = 0;
            } else {
                if (v < vmin) {
                    vmin = v;
                }
                if (v > vmax) {
                    vmax = v;
                }
            }
            sum += v;
            sumsq += v * v;
            count++;
        }
        if (count > 0) {
            mean = sum / (double)count;
            if (count > 1) {
                std = sqrt((sumsq - sum * sum / (double)count) / (double)(count - 1));
            }
        }
    }

    if (isnan(midpoint)) {
        midpoint = 0.5 * (vmin + vmax);
    }
    if (isnan(spread)) {
        spread = (kind == RESCALE_GAUSSIAN || kind == RESCALE_NEAR) ? 0.1 : 5.0;
    }
    if (isnan(minimum)) {
        minimum = vmin;
    }
    if (isnan(maximum)) {
        maximum = vmax;
    }
    if (isnan(shift)) {
        shift = vmin;
    }
    if (isnan(factor)) {
        factor = 1.0;
    }
    if (isnan(lower_threshold)) {
        lower_threshold = vmin;
    }
    if (isnan(upper_threshold)) {
        upper_threshold = vmax;
    }

    if (kind == RESCALE_GAUSSIAN || kind == RESCALE_NEAR || kind == RESCALE_LARGE || kind == RESCALE_SMALL) {
        p0 = midpoint;
        p1 = spread;
    } else if (kind == RESCALE_MSLARGE || kind == RESCALE_MSSMALL) {
        p0 = mean_mult;
        p1 = std_mult;
    } else if (kind == RESCALE_LINEAR) {
        p0 = minimum;
        p1 = maximum;
    } else if (kind == RESCALE_EXPONENTIAL || kind == RESCALE_LOGARITHM || kind == RESCALE_POWER) {
        p0 = shift;
        p1 = factor;
    } else if (kind == RESCALE_LOGISTIC_GROWTH || kind == RESCALE_LOGISTIC_DECAY) {
        p0 = minimum;
        p1 = maximum;
        p2 = y_intercept;
    } else if (kind == RESCALE_SYMMETRIC_LINEAR) {
        p0 = midpoint;
        p1 = minimum;
        p2 = maximum;
    }

    out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(arr), PyArray_DIMS(arr), NPY_DOUBLE);
    if (!out) {
        Py_DECREF(arr);
        return NULL;
    }

    /* first pass: raw transform, track range for non-[0,1] functions */
    for (i = 0; i < n; ++i) {
        double v = ((double *)PyArray_DATA(arr))[i];
        double u;
        if (is_nodata(v, nodata, has_nodata)) {
            ((double *)PyArray_DATA(out))[i] = NAN;
            continue;
        }
        if (v < lower_threshold) {
            u = from_scale;
            ((double *)PyArray_DATA(out))[i] = u;
            continue;
        }
        if (v > upper_threshold) {
            u = to_scale;
            ((double *)PyArray_DATA(out))[i] = u;
            continue;
        }
        u = rescale_unit(kind, v, p0, p1, p2, mean, std);
        ((double *)PyArray_DATA(out))[i] = u;
        if (!isnan(u)) {
            if (u < umin) {
                umin = u;
            }
            if (u > umax) {
                umax = u;
            }
        }
    }

    /* scale to from_scale..to_scale */
    for (i = 0; i < n; ++i) {
        double v = ((double *)PyArray_DATA(arr))[i];
        double u = ((double *)PyArray_DATA(out))[i];
        if (is_nodata(v, nodata, has_nodata)) {
            ((double *)PyArray_DATA(out))[i] = has_nodata ? nodata : NAN;
            continue;
        }
        if (v < lower_threshold || v > upper_threshold) {
            continue; /* already assigned */
        }
        if (kind <= RESCALE_LINEAR || kind == RESCALE_LOGISTIC_GROWTH || kind == RESCALE_LOGISTIC_DECAY
            || kind == RESCALE_SYMMETRIC_LINEAR) {
            /* already in ~[0,1] */
            ((double *)PyArray_DATA(out))[i] = from_scale + clamp01(u) * (to_scale - from_scale);
        } else {
            if (umax == umin) {
                ((double *)PyArray_DATA(out))[i] = from_scale;
            } else {
                ((double *)PyArray_DATA(out))[i] =
                    from_scale + ((u - umin) / (umax - umin)) * (to_scale - from_scale);
            }
        }
    }

    Py_DECREF(arr);
    return (PyObject *)out;
}

/* -------------------------------------------------------------------------- */
/* weighted_sum                                                               */
/* -------------------------------------------------------------------------- */

static PyObject *
suit_weighted_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"arrays", "weights", "nodata", NULL};
    PyObject *arrays_obj = NULL;
    PyObject *weights_obj = NULL;
    PyObject *nodata_obj = Py_None;
    PyObject *seq = NULL;
    PyArrayObject *weights = NULL;
    PyArrayObject *out = NULL;
    PyArrayObject **arrays = NULL;
    Py_ssize_t n_layers, k;
    npy_intp n = 0, i;
    double nodata = 0.0;
    int has_nodata = 0;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &arrays_obj, &weights_obj, &nodata_obj)) {
        return NULL;
    }

    seq = PySequence_Fast(arrays_obj, "arrays must be a sequence of rasters");
    if (!seq) {
        return NULL;
    }
    n_layers = PySequence_Fast_GET_SIZE(seq);
    if (n_layers < 1) {
        PyErr_SetString(PyExc_ValueError, "arrays must contain at least one layer");
        goto fail;
    }

    weights = (PyArrayObject *)PyArray_FROM_OTF(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!weights) {
        goto fail;
    }
    if (PyArray_SIZE(weights) != n_layers) {
        PyErr_SetString(PyExc_ValueError, "weights length must match number of arrays");
        goto fail;
    }

    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
        has_nodata = 1;
    }

    arrays = (PyArrayObject **)PyMem_Calloc((size_t)n_layers, sizeof(PyArrayObject *));
    if (!arrays) {
        PyErr_NoMemory();
        goto fail;
    }

    for (k = 0; k < n_layers; ++k) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, k);
        arrays[k] = (PyArrayObject *)PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!arrays[k]) {
            goto fail;
        }
        if (k == 0) {
            n = PyArray_SIZE(arrays[k]);
            out = (PyArrayObject *)PyArray_ZEROS(PyArray_NDIM(arrays[k]), PyArray_DIMS(arrays[k]), NPY_DOUBLE, 0);
            if (!out) {
                goto fail;
            }
        } else if (PyArray_SIZE(arrays[k]) != n) {
            PyErr_SetString(PyExc_ValueError, "all arrays must have the same shape");
            goto fail;
        }
    }

    for (i = 0; i < n; ++i) {
        double acc = 0.0;
        int any_nodata = 0;
        for (k = 0; k < n_layers; ++k) {
            double v = ((double *)PyArray_DATA(arrays[k]))[i];
            double w = ((double *)PyArray_DATA(weights))[k];
            if (is_nodata(v, nodata, has_nodata)) {
                any_nodata = 1;
                break;
            }
            acc += v * w;
        }
        ((double *)PyArray_DATA(out))[i] = any_nodata ? (has_nodata ? nodata : NAN) : acc;
    }

    for (k = 0; k < n_layers; ++k) {
        Py_XDECREF(arrays[k]);
    }
    PyMem_Free(arrays);
    Py_DECREF(weights);
    Py_DECREF(seq);
    return (PyObject *)out;

fail:
    if (arrays) {
        for (k = 0; k < n_layers; ++k) {
            Py_XDECREF(arrays[k]);
        }
        PyMem_Free(arrays);
    }
    Py_XDECREF(weights);
    Py_XDECREF(out);
    Py_XDECREF(seq);
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* weighted_overlay                                                           */
/* -------------------------------------------------------------------------- */

/*
 * weighted_overlay(arrays, weights, evaluation_scales=None, nodata=None)
 * arrays already remapped to a common evaluation scale, OR evaluation_scales
 * is a sequence of (n_rules, 3) rule arrays applied per layer first.
 * Result = round(sum(layer_i * weight_i) / sum(weights))
 */
static PyObject *
suit_weighted_overlay(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"arrays", "weights", "nodata", NULL};
    PyObject *arrays_obj = NULL;
    PyObject *weights_obj = NULL;
    PyObject *nodata_obj = Py_None;
    PyObject *seq = NULL;
    PyArrayObject *weights = NULL;
    PyArrayObject *out = NULL;
    PyArrayObject **arrays = NULL;
    Py_ssize_t n_layers, k;
    npy_intp n = 0, i;
    double nodata = 0.0;
    int has_nodata = 0;
    double wsum = 0.0;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &arrays_obj, &weights_obj, &nodata_obj)) {
        return NULL;
    }

    seq = PySequence_Fast(arrays_obj, "arrays must be a sequence of rasters");
    if (!seq) {
        return NULL;
    }
    n_layers = PySequence_Fast_GET_SIZE(seq);
    if (n_layers < 1) {
        PyErr_SetString(PyExc_ValueError, "arrays must contain at least one layer");
        goto fail;
    }

    weights = (PyArrayObject *)PyArray_FROM_OTF(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!weights) {
        goto fail;
    }
    if (PyArray_SIZE(weights) != n_layers) {
        PyErr_SetString(PyExc_ValueError, "weights length must match number of arrays");
        goto fail;
    }
    for (k = 0; k < n_layers; ++k) {
        wsum += ((double *)PyArray_DATA(weights))[k];
    }
    if (wsum == 0.0) {
        PyErr_SetString(PyExc_ValueError, "sum of weights must be non-zero");
        goto fail;
    }

    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
        has_nodata = 1;
    }

    arrays = (PyArrayObject **)PyMem_Calloc((size_t)n_layers, sizeof(PyArrayObject *));
    if (!arrays) {
        PyErr_NoMemory();
        goto fail;
    }

    for (k = 0; k < n_layers; ++k) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, k);
        arrays[k] = (PyArrayObject *)PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!arrays[k]) {
            goto fail;
        }
        if (k == 0) {
            n = PyArray_SIZE(arrays[k]);
            out = (PyArrayObject *)PyArray_SimpleNew(
                PyArray_NDIM(arrays[k]), PyArray_DIMS(arrays[k]), NPY_DOUBLE);
            if (!out) {
                goto fail;
            }
        } else if (PyArray_SIZE(arrays[k]) != n) {
            PyErr_SetString(PyExc_ValueError, "all arrays must have the same shape");
            goto fail;
        }
    }

    for (i = 0; i < n; ++i) {
        double acc = 0.0;
        int any_nodata = 0;
        for (k = 0; k < n_layers; ++k) {
            double v = ((double *)PyArray_DATA(arrays[k]))[i];
            double w = ((double *)PyArray_DATA(weights))[k];
            if (is_nodata(v, nodata, has_nodata)) {
                any_nodata = 1;
                break;
            }
            acc += v * w;
        }
        if (any_nodata) {
            ((double *)PyArray_DATA(out))[i] = has_nodata ? nodata : NAN;
        } else {
            ((double *)PyArray_DATA(out))[i] = floor(acc / wsum + 0.5);
        }
    }

    for (k = 0; k < n_layers; ++k) {
        Py_XDECREF(arrays[k]);
    }
    PyMem_Free(arrays);
    Py_DECREF(weights);
    Py_DECREF(seq);
    return (PyObject *)out;

fail:
    if (arrays) {
        for (k = 0; k < n_layers; ++k) {
            Py_XDECREF(arrays[k]);
        }
        PyMem_Free(arrays);
    }
    Py_XDECREF(weights);
    Py_XDECREF(out);
    Py_XDECREF(seq);
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* fuzzy_overlay                                                              */
/* -------------------------------------------------------------------------- */

static PyObject *
suit_fuzzy_overlay(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"arrays", "overlay_type", "gamma", "nodata", NULL};
    PyObject *arrays_obj = NULL;
    const char *overlay_type = "and";
    double gamma = 0.9;
    PyObject *nodata_obj = Py_None;
    PyObject *seq = NULL;
    PyArrayObject *out = NULL;
    PyArrayObject **arrays = NULL;
    Py_ssize_t n_layers, k;
    npy_intp n = 0, i;
    int kind;
    double nodata = 0.0;
    int has_nodata = 0;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|sdO", kwlist, &arrays_obj, &overlay_type, &gamma, &nodata_obj)) {
        return NULL;
    }

    kind = parse_overlay_type(overlay_type);
    if (kind < 0) {
        PyErr_Format(PyExc_ValueError, "Unknown overlay_type %s", overlay_type);
        return NULL;
    }
    if (kind == OVERLAY_GAMMA && (gamma < 0.0 || gamma > 1.0)) {
        PyErr_SetString(PyExc_ValueError, "gamma must be between 0 and 1");
        return NULL;
    }

    seq = PySequence_Fast(arrays_obj, "arrays must be a sequence of rasters");
    if (!seq) {
        return NULL;
    }
    n_layers = PySequence_Fast_GET_SIZE(seq);
    if (n_layers < 1) {
        PyErr_SetString(PyExc_ValueError, "arrays must contain at least one layer");
        goto fail;
    }

    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
        has_nodata = 1;
    }

    arrays = (PyArrayObject **)PyMem_Calloc((size_t)n_layers, sizeof(PyArrayObject *));
    if (!arrays) {
        PyErr_NoMemory();
        goto fail;
    }

    for (k = 0; k < n_layers; ++k) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, k);
        arrays[k] = (PyArrayObject *)PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!arrays[k]) {
            goto fail;
        }
        if (k == 0) {
            n = PyArray_SIZE(arrays[k]);
            out = (PyArrayObject *)PyArray_SimpleNew(
                PyArray_NDIM(arrays[k]), PyArray_DIMS(arrays[k]), NPY_DOUBLE);
            if (!out) {
                goto fail;
            }
        } else if (PyArray_SIZE(arrays[k]) != n) {
            PyErr_SetString(PyExc_ValueError, "all arrays must have the same shape");
            goto fail;
        }
    }

    for (i = 0; i < n; ++i) {
        int any_nodata = 0;
        double vmin = 0.0, vmax = 0.0, prod = 1.0, sum_comp = 1.0;
        for (k = 0; k < n_layers; ++k) {
            double v = ((double *)PyArray_DATA(arrays[k]))[i];
            if (is_nodata(v, nodata, has_nodata)) {
                any_nodata = 1;
                break;
            }
            v = clamp01(v);
            if (k == 0) {
                vmin = vmax = v;
            } else {
                if (v < vmin) {
                    vmin = v;
                }
                if (v > vmax) {
                    vmax = v;
                }
            }
            prod *= v;
            sum_comp *= (1.0 - v);
        }
        if (any_nodata) {
            ((double *)PyArray_DATA(out))[i] = has_nodata ? nodata : NAN;
            continue;
        }
        switch (kind) {
        case OVERLAY_AND:
            ((double *)PyArray_DATA(out))[i] = vmin;
            break;
        case OVERLAY_OR:
            ((double *)PyArray_DATA(out))[i] = vmax;
            break;
        case OVERLAY_PRODUCT:
            ((double *)PyArray_DATA(out))[i] = clamp01(prod);
            break;
        case OVERLAY_SUM:
            ((double *)PyArray_DATA(out))[i] = clamp01(1.0 - sum_comp);
            break;
        case OVERLAY_GAMMA: {
            double fsum = clamp01(1.0 - sum_comp);
            double fprod = clamp01(prod);
            double g = pow(fsum, gamma) * pow(fprod, 1.0 - gamma);
            ((double *)PyArray_DATA(out))[i] = clamp01(g);
            break;
        }
        default:
            ((double *)PyArray_DATA(out))[i] = NAN;
        }
    }

    for (k = 0; k < n_layers; ++k) {
        Py_XDECREF(arrays[k]);
    }
    PyMem_Free(arrays);
    Py_DECREF(seq);
    return (PyObject *)out;

fail:
    if (arrays) {
        for (k = 0; k < n_layers; ++k) {
            Py_XDECREF(arrays[k]);
        }
        PyMem_Free(arrays);
    }
    Py_XDECREF(out);
    Py_XDECREF(seq);
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* locate_regions                                                             */
/* -------------------------------------------------------------------------- */

typedef struct {
    npy_intp idx;
    double value;
} SuitCell;

static int
suit_cell_cmp_desc(const void *a, const void *b)
{
    const SuitCell *ca = (const SuitCell *)a;
    const SuitCell *cb = (const SuitCell *)b;
    if (ca->value < cb->value) {
        return 1;
    }
    if (ca->value > cb->value) {
        return -1;
    }
    if (ca->idx < cb->idx) {
        return -1;
    }
    if (ca->idx > cb->idx) {
        return 1;
    }
    return 0;
}

static double
shape_penalty(const char *shape, npy_intp r, npy_intp c, npy_intp sr, npy_intp sc, double target_cells)
{
    double dr = (double)(r - sr);
    double dc = (double)(c - sc);
    double dist2 = dr * dr + dc * dc;
    double radius;
    if (!shape) {
        return 0.0;
    }
    radius = sqrt(target_cells / M_PI);
    if (strcmp(shape, "circle") == 0) {
        if (dist2 > radius * radius * 1.21) {
            return 0.25;
        }
        return 1.0;
    }
    if (strcmp(shape, "ellipse") == 0) {
        /* elongated along rows */
        double a = radius * 1.6;
        double b = radius * 0.7;
        double v = (dc * dc) / (a * a) + (dr * dr) / (b * b);
        return (v <= 1.21) ? 1.0 : 0.25;
    }
    if (strcmp(shape, "square") == 0) {
        double half = 0.5 * sqrt(target_cells);
        double adr = fabs(dr);
        double adc = fabs(dc);
        return (adr <= half * 1.1 && adc <= half * 1.1) ? 1.0 : 0.25;
    }
    return 1.0;
}

/*
 * locate_regions(array, n_regions, region_size, shape='circle',
 *                minimum_distance=0, nodata=None)
 * region_size is in cells. Returns int32 labels (0=background).
 */
static PyObject *
suit_locate_regions(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "array", "n_regions", "region_size", "shape", "minimum_distance", "nodata", NULL};
    PyObject *array_obj = NULL;
    int n_regions = 1;
    int region_size = 1;
    const char *shape = "circle";
    double minimum_distance = 0.0;
    PyObject *nodata_obj = Py_None;
    PyArrayObject *arr = NULL;
    PyArrayObject *out = NULL;
    npy_intp rows, cols, n, i;
    double nodata = 0.0;
    int has_nodata = 0;
    SuitCell *order = NULL;
    npy_intp n_valid = 0;
    int *labels = NULL;
    int *claimed = NULL;
    double *centers_r = NULL;
    double *centers_c = NULL;
    int region_id;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Oii|sdO",
            kwlist,
            &array_obj,
            &n_regions,
            &region_size,
            &shape,
            &minimum_distance,
            &nodata_obj)) {
        return NULL;
    }

    if (n_regions < 1 || region_size < 1) {
        PyErr_SetString(PyExc_ValueError, "n_regions and region_size must be >= 1");
        return NULL;
    }

    arr = (PyArrayObject *)PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) {
        return NULL;
    }
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "array must be 2-dimensional");
        Py_DECREF(arr);
        return NULL;
    }
    rows = PyArray_DIM(arr, 0);
    cols = PyArray_DIM(arr, 1);
    n = rows * cols;

    if (nodata_obj != Py_None) {
        nodata = PyFloat_AsDouble(nodata_obj);
        if (PyErr_Occurred()) {
            Py_DECREF(arr);
            return NULL;
        }
        has_nodata = 1;
    }

    out = (PyArrayObject *)PyArray_ZEROS(2, PyArray_DIMS(arr), NPY_INT32, 0);
    if (!out) {
        Py_DECREF(arr);
        return NULL;
    }

    order = (SuitCell *)PyMem_Malloc((size_t)n * sizeof(SuitCell));
    labels = (int *)PyMem_Calloc((size_t)n, sizeof(int));
    claimed = (int *)PyMem_Calloc((size_t)n, sizeof(int));
    centers_r = (double *)PyMem_Malloc((size_t)n_regions * sizeof(double));
    centers_c = (double *)PyMem_Malloc((size_t)n_regions * sizeof(double));
    if (!order || !labels || !claimed || !centers_r || !centers_c) {
        PyErr_NoMemory();
        goto fail;
    }

    for (i = 0; i < n; ++i) {
        npy_intp r = i / cols;
        npy_intp c = i % cols;
        double v = *((double *)PyArray_GETPTR2(arr, r, c));
        if (is_nodata(v, nodata, has_nodata)) {
            claimed[i] = 1;
            continue;
        }
        order[n_valid].idx = i;
        order[n_valid].value = v;
        n_valid++;
    }
    qsort(order, (size_t)n_valid, sizeof(SuitCell), suit_cell_cmp_desc);

    for (region_id = 1; region_id <= n_regions; ++region_id) {
        npy_intp seed = -1;
        npy_intp sr = 0, sc = 0;
        npy_intp grown = 0;
        npy_intp *frontier = NULL;
        npy_intp frontier_n = 0;
        npy_intp frontier_cap = 0;
        npy_intp oi;

        /* pick highest remaining seed respecting minimum distance */
        for (oi = 0; oi < n_valid; ++oi) {
            npy_intp idx = order[oi].idx;
            npy_intp r, c, prev;
            int ok = 1;
            if (claimed[idx]) {
                continue;
            }
            r = idx / cols;
            c = idx % cols;
            for (prev = 0; prev < region_id - 1; ++prev) {
                double dr = (double)r - centers_r[prev];
                double dc = (double)c - centers_c[prev];
                if (sqrt(dr * dr + dc * dc) < minimum_distance) {
                    ok = 0;
                    break;
                }
            }
            if (ok) {
                seed = idx;
                sr = r;
                sc = c;
                break;
            }
        }
        if (seed < 0) {
            break;
        }

        centers_r[region_id - 1] = (double)sr;
        centers_c[region_id - 1] = (double)sc;
        labels[seed] = region_id;
        claimed[seed] = 1;
        grown = 1;

        frontier_cap = 64;
        frontier = (npy_intp *)PyMem_Malloc((size_t)frontier_cap * sizeof(npy_intp));
        if (!frontier) {
            PyErr_NoMemory();
            goto fail;
        }
        frontier[frontier_n++] = seed;

        while (grown < region_size && frontier_n > 0) {
            /* pick best neighbor of any frontier cell */
            double best_score = -DBL_MAX;
            npy_intp best_idx = -1;
            npy_intp fi, d;
            const npy_intp drs[4] = {-1, 1, 0, 0};
            const npy_intp dcs[4] = {0, 0, -1, 1};

            for (fi = 0; fi < frontier_n; ++fi) {
                npy_intp base = frontier[fi];
                npy_intp br = base / cols;
                npy_intp bc = base % cols;
                for (d = 0; d < 4; ++d) {
                    npy_intp nr = br + drs[d];
                    npy_intp nc = bc + dcs[d];
                    npy_intp nidx;
                    double v, score, pen;
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                        continue;
                    }
                    nidx = nr * cols + nc;
                    if (claimed[nidx]) {
                        continue;
                    }
                    v = *((double *)PyArray_GETPTR2(arr, nr, nc));
                    if (is_nodata(v, nodata, has_nodata)) {
                        claimed[nidx] = 1;
                        continue;
                    }
                    pen = shape_penalty(shape, nr, nc, sr, sc, (double)region_size);
                    score = v * pen;
                    if (score > best_score) {
                        best_score = score;
                        best_idx = nidx;
                    }
                }
            }

            if (best_idx < 0) {
                break;
            }
            labels[best_idx] = region_id;
            claimed[best_idx] = 1;
            grown++;
            if (frontier_n + 1 > frontier_cap) {
                npy_intp new_cap = frontier_cap * 2;
                npy_intp *tmp = (npy_intp *)PyMem_Realloc(frontier, (size_t)new_cap * sizeof(npy_intp));
                if (!tmp) {
                    PyMem_Free(frontier);
                    PyErr_NoMemory();
                    goto fail;
                }
                frontier = tmp;
                frontier_cap = new_cap;
            }
            frontier[frontier_n++] = best_idx;
        }
        PyMem_Free(frontier);
    }

    for (i = 0; i < n; ++i) {
        npy_intp r = i / cols;
        npy_intp c = i % cols;
        *((npy_int32 *)PyArray_GETPTR2(out, r, c)) = (npy_int32)labels[i];
    }

    PyMem_Free(order);
    PyMem_Free(labels);
    PyMem_Free(claimed);
    PyMem_Free(centers_r);
    PyMem_Free(centers_c);
    Py_DECREF(arr);
    return (PyObject *)out;

fail:
    PyMem_Free(order);
    PyMem_Free(labels);
    PyMem_Free(claimed);
    PyMem_Free(centers_r);
    PyMem_Free(centers_c);
    Py_XDECREF(out);
    Py_DECREF(arr);
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* module                                                                     */
/* -------------------------------------------------------------------------- */

static PyMethodDef SuitabilityMethods[] = {
    {"reclassify", (PyCFunction)suit_reclassify, METH_VARARGS | METH_KEYWORDS, "Shift/reclass array values."},
    {"fuzzy_membership",
     (PyCFunction)suit_fuzzy_membership,
     METH_VARARGS | METH_KEYWORDS,
     "Fuzzy membership transform."},
    {"rescale_by_function",
     (PyCFunction)suit_rescale_by_function,
     METH_VARARGS | METH_KEYWORDS,
     "Rescale array by a transformation function."},
    {"weighted_sum", (PyCFunction)suit_weighted_sum, METH_VARARGS | METH_KEYWORDS, "Weighted sum of arrays."},
    {"weighted_overlay",
     (PyCFunction)suit_weighted_overlay,
     METH_VARARGS | METH_KEYWORDS,
     "Weighted overlay of evaluation-scale arrays."},
    {"fuzzy_overlay", (PyCFunction)suit_fuzzy_overlay, METH_VARARGS | METH_KEYWORDS, "Fuzzy overlay operators."},
    {"locate_regions",
     (PyCFunction)suit_locate_regions,
     METH_VARARGS | METH_KEYWORDS,
     "Locate discrete candidate regions on a suitability surface."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef suitability_module = {
    PyModuleDef_HEAD_INIT,
    "geolattice.spatial_analysis._suitability",
    "C kernels for suitability analysis (reclassify, fuzzy, weighted overlay, locate regions).",
    -1,
    SuitabilityMethods};

PyMODINIT_FUNC
PyInit__suitability(void)
{
    PyObject *m;
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    m = PyModule_Create(&suitability_module);
    return m;
}
