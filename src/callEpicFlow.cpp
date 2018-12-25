#include <stdlib.h>
#include <string.h>
#include "epic.h"
#include "image.h"
#include "io.h"
#include "variational.h"
#include<string>
#include"CVX/core.h"
using namespace cv;

Mat2f callEpicFlow(const std::string &srcfile, const std::string &tarfile, const std::string &sedEdgeFile, const std::string &matchFile, const std::string &datasetName)
{
	// read arguments
	color_image_t *im1 = color_image_load(srcfile.c_str());
	color_image_t *im2 = color_image_load(tarfile.c_str());
	float_image edges = read_edges(sedEdgeFile.c_str(), im1->width, im1->height);
	float_image matches = read_matches(matchFile.c_str());
	//const char *outputfile = argv[5];

	// prepare variables
	epic_params_t epic_params;
	epic_params_default(&epic_params);
	variational_params_t flow_params;
	variational_params_default(&flow_params);
	image_t *wx = image_new(im1->width, im1->height), *wy = image_new(im1->width, im1->height);

	// read optional arguments 

		if (datasetName=="sintel") {
			epic_params.pref_nn = 25;
			epic_params.nn = 160;
			epic_params.coef_kernel = 1.1f;
			flow_params.niter_outer = 5;
			flow_params.alpha = 1.0f;
			flow_params.gamma = 0.72f;
			flow_params.delta = 0.0f;
			flow_params.sigma = 1.1f;
		}
		else if (datasetName == "kitti") {
			epic_params.pref_nn = 25;
			epic_params.nn = 160;
			epic_params.coef_kernel = 1.1f;
			flow_params.niter_outer = 2;
			flow_params.alpha = 1.0f;
			flow_params.gamma = 0.77f;
			flow_params.delta = 0.0f;
			flow_params.sigma = 1.7f;
		}
		else if (datasetName == "middlebury") {
			epic_params.pref_nn = 15;
			epic_params.nn = 65;
			epic_params.coef_kernel = 0.2f;
			flow_params.niter_outer = 25;
			flow_params.alpha = 1.0f;
			flow_params.gamma = 0.72f;
			flow_params.delta = 0.0f;
			flow_params.sigma = 1.1f;
		}
		else {
			fprintf(stderr, "unknown argument %s", datasetName.c_str());
			exit(1);
		}
	

	// compute interpolation and energy minimization
	color_image_t *imlab = rgb_to_lab(im1);
	epic(wx, wy, imlab, &matches, &edges, &epic_params, 1);
	// energy minimization
	variational(wx, wy, im1, im2, &flow_params);
	// write output file and free memory
	//writeFlowFile(outputfile, wx, wy);

	Mat2f flow(im1->height, im1->width);
	for_each_3c(DWHNC(flow), wx->data, wx->stride, ccn1(), wy->data, wy->stride, ccn1(), [](float *p, float dx, float dy, int x, int y) {
		p[0] = x + dx;
		p[1] = y + dy;
	});

	color_image_delete(im1);
	color_image_delete(imlab);
	color_image_delete(im2);
	free(matches.pixels);
	free(edges.pixels);
	image_delete(wx);
	image_delete(wy);

	return flow;
}
