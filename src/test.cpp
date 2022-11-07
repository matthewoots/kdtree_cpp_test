#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <chrono>
#include <Eigen/Dense>
#include <vector>
#include "kdtree.h"

using namespace std;
using namespace std::chrono;

kdtree *kd;
kdres *set;

int main(int argc)
{
	int i, rounds = 10, vcount = 10;
    std::vector<Eigen::Vector3d> r;
    for (i = 0; i < rounds; i++)
    {
        r.push_back(Eigen::Vector3d(
            (rand() / RAND_MAX) * 200.0 - 100.0,
            (rand() / RAND_MAX) * 200.0 - 100.0,
            (rand() / RAND_MAX) * 200.0 - 100.0
        ));
    }
	printf("inserting %d random vectors... \n", vcount);
	// fflush(stdout);

    for (Eigen::Vector3d &p : r)
    {
        kd_free(kd);
        kd = kd_create(3);

        time_point<std::chrono::system_clock> start = system_clock::now();
        for (i = 0; i < vcount; i++) {
            float x, y, z;
            x = ((float)rand() / RAND_MAX) * 200.0 - 100.0;
            y = ((float)rand() / RAND_MAX) * 200.0 - 100.0;
            z = ((float)rand() / RAND_MAX) * 200.0 - 100.0;
            printf("%d. (%f, %f, %f)\n", i, x, y, z);
            int v = kd_insert3(kd, x, y, z, 0);
        }
        double msec = duration<double>(system_clock::now() - start).count()*1000;
        printf("%.3f ms\n", (float)msec);

        start = system_clock::now();
        set = kd_nearest_range3(kd, p.x(), p.y(), p.z(), 100);
        msec = duration<double>(system_clock::now() - start).count()*1000;
        printf("range query returned %d items in %.5f ms\n", kd_res_size(set), (float)msec);
        kd_res_free(set);
    }
	kd_free(kd);
	return 0;
}
