#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double quadratic(double a, double b, double c)
{
    double r1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
    double r2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
    if (r1 > 0 && r2 > 0)
        return r1 < r2 ? r1 : r2;
    if (r1 > 0)
        return r1;
    return r2;
}

double citardauq(double a, double b, double c)
{
    double d = sqrt(b * b - 4 * a * c);
    double t = 0;
    if (b >= 0)
    {
        t = -b - d;
    }
    else
    {
        t = -b + d;
    }
    double r1 = t / (2 * a);
    double r2 = (2 * c) / t;
    if (r1 > 0 && r2 > 0)
        return r1 < r2 ? r1 : r2;
    if (r1 > 0)
        return r1;
    return r2;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s a b c\n", argv[0]);
        return 1;
    }
    double a = atof(argv[1]);
    double b = atof(argv[2]);
    double c = atof(argv[3]);
    printf("%.17g %.17g\n", quadratic(a, b, c), citardauq(a, b, c));
    return 0;
}
