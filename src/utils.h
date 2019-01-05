#pragma once

#include <iostream>
#include <iomanip>
#include <random>
#include "tensor.h"


namespace hen {
// random function
void Randomfunction(float data[], int n, int seed, float bound);

void PrintSample(FloatTensor &tensor);

int CountCorrect(FloatTensor &output, FloatTensor &target);

}