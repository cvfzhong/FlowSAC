
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<time.h>
#include "epic.h"
#include "image.h"
#include "array_types.h"
#include "epic_aux.h"

#include "omp.h"

//#include "io.h"

/* create a copy of input matches with 4 columns, with all points inside the image area*/
static float_image rectify_corres(const float_image* matches, const int w1, const int h1, const int w2, const int h2, const int n_thread) {
	float_image res = empty_image(float, 4, matches->ty);
	int i;
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
	for (i = 0; i<matches->ty; i++) {
		res.pixels[4 * i] = MAX(0, MIN(matches->pixels[i*matches->tx], w1 - 1));
		res.pixels[4 * i + 1] = MAX(0, MIN(matches->pixels[i*matches->tx + 1], h1 - 1));
		res.pixels[4 * i + 2] = MAX(0, MIN(matches->pixels[i*matches->tx + 2], w2 - 1));
		res.pixels[4 * i + 3] = MAX(0, MIN(matches->pixels[i*matches->tx + 3], h2 - 1));
	}
	return res;
}

/* given a set of matches, return the set of points in the first image where a match exists */
static int_image matches_to_seeds(const float_image *matches, const int n_thread) {
	int_image res = empty_image(int, 2, matches->ty);
	int i;
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
	for (i = 0; i<matches->ty; i++) {
		res.pixels[2 * i] = (int)matches->pixels[4 * i];
		res.pixels[2 * i + 1] = (int)matches->pixels[4 * i + 1];
	}
	return res;
}

/* given a set of matches, return the set of vecotrs of the matches*/
static float_image matches_to_vects(const float_image *matches, const int n_thread) {
	float_image res = empty_image(float, 2, matches->ty);
	int i;
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
	for (i = 0; i<matches->ty; i++) {
		res.pixels[2 * i] = matches->pixels[4 * i + 2] - matches->pixels[4 * i];
		res.pixels[2 * i + 1] = matches->pixels[4 * i + 3] - matches->pixels[4 * i + 1];
	}
	return res;
}

/* remove matches coming from a pixel with a low saliency */
static void apply_saliency_threshold(float_image *matches, const color_image_t *im, const float saliency_threshold) {
	image_t *s = saliency(im, 0.8f, 1.0f);
	float_image tmp = empty_image(float, 4, matches->ty);
	int i, ii = 0;
	for (i = 0; i<matches->ty; i++) {
		if (s->data[(int)(matches->pixels[i * 4 + 1] * s->stride + matches->pixels[i * 4])] >= saliency_threshold) {
			memcpy(&tmp.pixels[ii * 4], &matches->pixels[i * 4], sizeof(float) * 4);
			ii += 1;
		}
	}
	image_delete(s);
	REALLOC(tmp.pixels, float, ii * 4);
	matches->ty = ii;
	free(matches->pixels);
	matches->pixels = tmp.pixels;
}

/* remove matches where the nadaraya-watson estimation is too different from the input match */
static void prefiltering(float_image *matches, const float_image *edges, const int nn, const float threshold, const float coef_kernel, const int n_thread) {
	const float th2 = threshold*threshold;
	const int nns = MIN(nn + 1, matches->ty); // nn closest plus itself
	if (nns != nn + 1) fprintf(stderr, "Warning: not enough matches for prefiltering\n");
	int_image seeds = matches_to_seeds(matches, n_thread);
	float_image vects = matches_to_vects(matches, n_thread);

	// compute closest matches
	int_image nnf = empty_image(int, nns, matches->ty);
	float_image dis = empty_image(float, nns, matches->ty);
	int_image labels = empty_image(int, edges->tx, edges->ty);
	dist_trf_nnfield_subset(&nnf, &dis, &labels, &seeds, edges, NULL, &seeds, n_thread);

	// apply kernel to the distance
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
	for (int i = 0; i<dis.tx*dis.ty; i++) {
		dis.pixels[i] = expf(-coef_kernel*dis.pixels[i]) + 1e-08;
	}

	// compute nadaraya-watson estimation
	float_image seedsvects = empty_image(float, 2, matches->ty);
	fit_nadarayawatson(&seedsvects, &nnf, &dis, &vects, n_thread);

	// remove matches if necessary
	float_image tmp = empty_image(float, 4, matches->ty);
	int ii = 0;
	for (int i = 0; i<matches->ty; i++) {
		if (pow2(seedsvects.pixels[2 * i] - vects.pixels[2 * i]) + pow2(seedsvects.pixels[2 * i + 1] - vects.pixels[2 * i + 1])<th2) {
			memcpy(&tmp.pixels[ii * 4], &matches->pixels[i * 4], sizeof(float) * 4);
			ii += 1;
		}
	}
	REALLOC(tmp.pixels, float, ii * 4);
	matches->ty = ii;
	free(matches->pixels);
	matches->pixels = tmp.pixels;

	// free memory
	free(seeds.pixels);
	free(vects.pixels);
	free(nnf.pixels);
	free(dis.pixels);
	free(labels.pixels);
	free(seedsvects.pixels);
}


/* set params to default value */
static void epic_params_default2(epic_params_t* params) {
	strcpy(params->method, "LA");
	params->saliency_th = 0.045f;
	params->pref_nn = 25;
	params->pref_th = 5.0f;
	params->nn = 100;
	params->coef_kernel = 0.8f;
	params->euc = 0.001f;
	params->verbose = 0;
}

/* main function for edge-preserving interpolation of correspondences
flowx                  x-component of the flow (output)
flowy                  y-component of the flow (output)
input_matches          input matches with each line a match and the first four columns containing x1 y1 x2 y2
im                     first image (in lab colorspace)
edges                  edges cost (can be modified)
params                 parameters
n_thread               number of threads
*/
static void epic_no_filter(image_t *flowx, image_t *flowy, const color_image_t *im, const float_image *input_matches, float_image* edges, const epic_params_t* params, const int n_thread) {

	// copy matches and correct them if necessary
	float_image matches = rectify_corres(input_matches, im->width, im->height, im->width, im->height, n_thread);
	if (params->verbose) printf("%d input matches\n", matches.ty);




	// saliency filter
	//if (params->saliency_th) {
	//	apply_saliency_threshold(&matches, im, params->saliency_th);
	//	if (params->verbose) printf("Saliency filtering, remaining %d matches\n", matches.ty);
	//}
	//// consistency filter
	//if (params->pref_nn) {
	//	prefiltering(&matches, edges, params->pref_nn, params->pref_th, params->coef_kernel, n_thread);
	//	if (params->verbose) printf("Consistenct filter, remaining %d matches\n", matches.ty);
	//}
	//time_t beg = clock();
	// prepare variables
	const int nns = MIN(params->nn, matches.ty);
	if (nns < params->nn) fprintf(stderr, "Warning: not enough matches for interpolating\n");
	if (params->verbose) printf("Computing %d nearest neighbors for each match\n", nns);
	int_image seeds = matches_to_seeds(&matches, n_thread);
	float_image vects = matches_to_vects(&matches, n_thread);

	// compute nearest matches for each seed
	int_image nnf = empty_image(int, nns, matches.ty);
	float_image dis = empty_image(float, nns, matches.ty);
	int_image labels = empty_image(int, edges->tx, edges->ty);
	dist_trf_nnfield_subset(&nnf, &dis, &labels, &seeds, edges, NULL, &seeds, n_thread);

	//printf("T1=%d\n", int(clock() - beg));
	// apply kernel to the distance
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
	for (int i = 0; i<dis.tx*dis.ty; i++) {
		dis.pixels[i] = expf(-params->coef_kernel*dis.pixels[i]) + 1e-08;
	}

	// interpolation
	if (params->verbose) printf("Interpolation of matches using %s\n", params->method);
	float_image newvects = empty_image(float, 2, im->width*im->height);
	if (!strcmp(params->method, "LA")) {
	//	printf("T2=%d\n", int(clock() - beg));
		float_image seedsaffine = empty_image(float, 6, vects.ty);
		fit_localaffine(&seedsaffine, &nnf, &dis, &seeds, &vects);
	//	printf("T3=%d\n", int(clock() - beg));
		apply_localaffine(&newvects, &seedsaffine, &labels, n_thread);
	//	printf("T4=%d\n", int(clock() - beg));
		free(seedsaffine.pixels);
	}
	else if (!strcmp(params->method, "NW")) {
		float_image seedsvects = empty_image(float, 2, vects.ty);
		fit_nadarayawatson(&seedsvects, &nnf, &dis, &vects, n_thread);
		apply_nadarayawatson(&newvects, &seedsvects, &labels, n_thread);
		free(seedsvects.pixels);
	}
	else {
		fprintf(stderr, "method %s not recognized\n", params->method);
		exit(EXIT_FAILURE);
	}

	// copy result to the output
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
	for (int i = 0; i<im->height; i++) {
		for (int j = 0; j<im->width; j++) {
			flowx->data[i*im->stride + j] = newvects.pixels[2 * (i*im->width + j)];
			flowy->data[i*im->stride + j] = newvects.pixels[2 * (i*im->width + j) + 1];
		}
	}

	// free memory
	free(seeds.pixels);
	free(vects.pixels);
	free(nnf.pixels);
	free(dis.pixels);
	free(labels.pixels);
	free(newvects.pixels);
	free(matches.pixels);
}

#include"flowsac.h"
#include"variational.h"
#include"io.h"
#include"utils.h"
#include"BFC/err.h"

color_image_t *image_from(const Mat3b &img)
{
	color_image_t* image = color_image_new(img.cols, img.rows);
	const uchar *image_data = img.data;

	for (int y = 0; y < img.rows; ++y, image_data += img.step)
	{
		int j = image->stride*y;

		for (int x = 0; x < img.cols; ++x)
		{
			image->c1[j+x] = image_data[3 * x + 0];
			image->c2[j+x] = image_data[3 * x + 1];
			image->c3[j+x] = image_data[3 * x + 2];
		}
	}
	return image;
}

float_image image_from(const Mat1f &img) 
{
	float_image res = empty_image(float, img.cols, img.rows);
	CV_Assert(img.step == sizeof(float)*img.cols);
	memcpy(res.pixels, img.data, sizeof(float)*img.cols*img.rows);
	return res;
}

float_image image_from(const std::vector<Match> &matches) {
	float_image res = empty_image(float, 4, matches.size());

	for(int n=0; n<matches.size(); ++n) {
		auto &m = matches[n];
		res.pixels[4 * n] = m.x0;
		res.pixels[4 * n + 1] = m.y0;
		res.pixels[4 * n + 2] = m.x1;
		res.pixels[4 * n + 3] = m.y1;
	}
	return res;
}


//class EpicInterpDS
//	:public Interpolator
//{
//	color_image_t *_src, *_srcLab, *_tar;
//	float_image  _sedEdge;
//	epic_params_t _epicParams;
//	variational_params_t _flowParams;
//
//	color_image_t *_src2, *_srcLab2;
//	float_image    _sedEdge2;
//public:
//	~EpicInterpDS()
//	{
//		color_image_delete(_src);
//		color_image_delete(_srcLab);
//		color_image_delete(_tar);
//
//		free(_sedEdge.pixels);
//
//		color_image_delete(_src2);
//		color_image_delete(_srcLab2);
//		free(_sedEdge2.pixels);
//	}
//	void setParams(const std::string &datasetName)
//	{
//		epic_params_t epic_params;
//		epic_params_default(&epic_params);
//		variational_params_t flow_params;
//		variational_params_default(&flow_params);
//
//		if (stricmp(datasetName.c_str(), "sintel") == 0) {
//			epic_params.pref_nn = 25;
//			epic_params.nn = 160;
//			epic_params.coef_kernel = 1.1f;
//			flow_params.niter_outer = 5;
//			flow_params.alpha = 1.0f;
//			flow_params.gamma = 0.72f;
//			flow_params.delta = 0.0f;
//			flow_params.sigma = 1.1f;
//		}
//		else if (stricmp(datasetName.c_str(), "kitti") == 0) {
//			epic_params.pref_nn = 25;
//			epic_params.nn = 160;
//			epic_params.coef_kernel = 1.1f;
//			flow_params.niter_outer = 2;
//			flow_params.alpha = 1.0f;
//			flow_params.gamma = 0.77f;
//			flow_params.delta = 0.0f;
//			flow_params.sigma = 1.7f;
//		}
//		else if (stricmp(datasetName.c_str(), "middlebury") == 0) {
//			epic_params.pref_nn = 15;
//			epic_params.nn = 65;
//			epic_params.coef_kernel = 0.2f;
//			flow_params.niter_outer = 25;
//			flow_params.alpha = 1.0f;
//			flow_params.gamma = 0.72f;
//			flow_params.delta = 0.0f;
//			flow_params.sigma = 1.1f;
//		}
//		else
//			FF_EXCEPTION("","unknown dataset");
//
//		_epicParams = epic_params;
//		_flowParams = flow_params;
//	}
//	template<typename _MatT>
//	static _MatT downsample(const _MatT &img)
//	{
//#if 0
//		Size dsize(img.cols / 2, img.rows / 2);
//		_MatT dimg;
//		cv::resize(img, dimg, dsize, 0, 0, INTER_NEAREST);
//		return dimg;
//#else
//		return img;
//#endif
//	}
//	void set(const Mat3b &src, const std::string &srcFile, const Mat3b &tar, const std::string &tarFile, const Mat1f &sedEdge, const std::string &datasetName)
//	{
//#if 1
//		_src = image_from(src);
//		_tar = image_from(tar);
//#else
//		_src = color_image_load(srcFile.c_str());
//		_tar = color_image_load(tarFile.c_str());
//#endif
//		_sedEdge = image_from(sedEdge);
//
//		_srcLab = rgb_to_lab(_src);
//		setParams(datasetName);
//
//		//moved from the function epic(...)
//		// eventually add a constant to edges cost
//		if (_epicParams.euc) {
//			int i;
//#if defined(USE_OPENMP)
//#pragma omp parallel for num_threads(n_thread)
//#endif
//			for (i = 0; i<_sedEdge.tx*_sedEdge.ty; i++) {
//				_sedEdge.pixels[i] += _epicParams.euc;
//			}
//		}
//		_src2 = image_from(downsample(src));
//		_sedEdge2 = image_from(downsample(sedEdge));
//		_srcLab2 = rgb_to_lab(_src2);
//	
//		if (_epicParams.euc) {
//			int i;
//#if defined(USE_OPENMP)
//#pragma omp parallel for num_threads(n_thread)
//#endif
//			for (i = 0; i<_sedEdge2.tx*_sedEdge2.ty; i++) {
//				_sedEdge2.pixels[i] += _epicParams.euc;
//			}
//		}
//	}
//	Mat2f exec(const std::vector<Match> &vMatches, bool fast)
//	{
//		Size csize = !fast ? Size(_src->width, _src->height) : Size(_src2->width, _src2->height);
//		double fastScale = _src2->width / double(_src->width);
//
//		image_t *wx = image_new(csize.width,csize.height), *wy = image_new(csize.width, csize.height);
//		float_image matches;
//		Mat2f flow(csize);
//
//		//strcpy(_epicParams.method, "NW");
//
//		if (!fast)
//		{
//			matches = image_from(vMatches);
//
//			epic_no_filter(wx, wy, _srcLab, &matches, &_sedEdge, &_epicParams, 1);
//			variational(wx, wy, _src, _tar, &_flowParams);
//		}
//		else
//		{
//			std::vector<Match> dmatch(vMatches);
//			for (auto &m : dmatch)
//			{
//				m.x0 *= fastScale; m.y0 *= fastScale; m.x1 *= fastScale; m.y1 *=fastScale;
//			}
//			matches = image_from(dmatch);
//
//			epic_no_filter(wx, wy, _srcLab2, &matches, &_sedEdge2, &_epicParams, 1);
//		}
//
//		for_each_3c(DWHNC(flow), wx->data, wx->stride, ccn1(), wy->data, wy->stride, ccn1(), [](float *p, float dx, float dy, int x, int y) {
//			p[0] = x + dx;
//			p[1] = y + dy;
//		});
//
//		if (fast && _src->width!=_src2->width)
//		{
//			flow *= 1.0/fastScale;
//			resize(flow, flow, Size(_src->width, _src->height));
//		}
//
//		free(matches.pixels);
//		image_delete(wx);
//		image_delete(wy);
//
//
//		return flow;
//	}
//};

class EpicInterp
	:public Interpolator
{
	color_image_t *_src, *_srcLab, *_tar;
	float_image  _sedEdge;
	epic_params_t _epicParams;
	variational_params_t _flowParams;
public:
	~EpicInterp()
	{
		color_image_delete(_src);
		color_image_delete(_srcLab);
		color_image_delete(_tar);

		free(_sedEdge.pixels);
	}
	void setParams(const std::string &datasetName)
	{
		epic_params_t epic_params;
		epic_params_default2(&epic_params);
		variational_params_t flow_params;
		variational_params_default(&flow_params);

		if (stricmp(datasetName.c_str(), "sintel") == 0) {
			epic_params.pref_nn = 25;
			epic_params.nn = 160;
			epic_params.coef_kernel = 1.1f;
			flow_params.niter_outer = 5;
			flow_params.alpha = 1.0f;
			flow_params.gamma = 0.72f;
			flow_params.delta = 0.0f;
			flow_params.sigma = 1.1f;
		}
		else if (stricmp(datasetName.c_str(), "kitti") == 0) {
			epic_params.pref_nn = 25;
			epic_params.nn = 160;
			epic_params.coef_kernel = 1.1f;
			flow_params.niter_outer = 2;
			flow_params.alpha = 1.0f;
			flow_params.gamma = 0.77f;
			flow_params.delta = 0.0f;
			flow_params.sigma = 1.7f;
		}
		else if (stricmp(datasetName.c_str(), "middlebury") == 0) {
			epic_params.pref_nn = 15;
			epic_params.nn = 65;
			epic_params.coef_kernel = 0.2f;
			flow_params.niter_outer = 25;
			flow_params.alpha = 1.0f;
			flow_params.gamma = 0.72f;
			flow_params.delta = 0.0f;
			flow_params.sigma = 1.1f;
		}
		else
			FF_EXCEPTION("", "unknown dataset");

		_epicParams = epic_params;
		_flowParams = flow_params;
	}
	void set(const Mat3b &src, const std::string &srcFile, const Mat3b &tar, const std::string &tarFile, const Mat1f &sedEdge, const std::string &datasetName)
	{
#if 1
		_src = image_from(src);
		_tar = image_from(tar);
#else
		_src = color_image_load(srcFile.c_str());
		_tar = color_image_load(tarFile.c_str());
#endif
		_sedEdge = image_from(sedEdge);

		_srcLab = rgb_to_lab(_src);
		setParams(datasetName);

		//moved from the function epic(...)
		// eventually add a constant to edges cost
		if (_epicParams.euc) {
			int i;
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
			for (i = 0; i<_sedEdge.tx*_sedEdge.ty; i++) {
				_sedEdge.pixels[i] += _epicParams.euc;
			}
		}
	}
	Mat2f exec(const std::vector<Match> &vMatches, bool fast)
	{
		Size csize = Size(_src->width, _src->height);

		image_t *wx = image_new(csize.width, csize.height), *wy = image_new(csize.width, csize.height);
		float_image matches=image_from(vMatches);
		Mat2f flow(csize);

		epic_no_filter(wx, wy, _srcLab, &matches, &_sedEdge, &_epicParams, 1);

		if (!fast)
			variational(wx, wy, _src, _tar, &_flowParams);

		for_each_3c(DWHNC(flow), wx->data, wx->stride, ccn1(), wy->data, wy->stride, ccn1(), [](float *p, float dx, float dy, int x, int y) {
			p[0] = x + dx;
			p[1] = y + dy;
		});

		free(matches.pixels);
		image_delete(wx);
		image_delete(wy);


		return flow;
	}
};

Interpolator* createEpicInterpolator(const Mat3b &src, const std::string &srcFile, const Mat3b &tar, const std::string &tarFile, const Mat1f &sedEdge, const std::string &datasetName)
{
	EpicInterp *obj = new EpicInterp;
	obj->set(src, srcFile, tar, tarFile, sedEdge, datasetName);
	return obj;
}



#if 0
#include"epic_aux.cpp"
#include"image.c"
#include"variational.c"
#include"variational_aux.c"
#include"solver.c"
#endif
