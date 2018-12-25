#pragma once

#include"flowsac.h"

template<typename _FlowT>
inline void xy2offset(const Mat_<_FlowT> &u, const Mat_<_FlowT> &v, Mat_<_FlowT> &dx, Mat_<_FlowT> &dy)
{
	dx.create(u.size());
	dy.create(v.size());

	for_each_4c(DWHN1(u), DN1(v), DN1(dx), DN1(dy), [](_FlowT u, _FlowT v, _FlowT &dx, _FlowT &dy, int x, int y) {
		dx = u - x; dy = v - y;
	});
}

template<typename _FlowT>
inline void xy2offset(const Mat_<Vec<_FlowT, 2> > &uv, Mat_<Vec<_FlowT, 2> > &dxy)
{
	dxy.create(uv.size());

	for_each_2c(DWHNC(uv), DNC(dxy), [](const _FlowT *uv, _FlowT *dxy, int x, int y) {
		dxy[0] = uv[0] - x;
		dxy[1] = uv[1] - y;
	});
}

template<typename _FlowT>
inline void xy2offset(Mat_<_FlowT> &u, Mat_<_FlowT> &v)
{
	for_each_2c(DWHN1(u), DN1(v), [](_FlowT &u, _FlowT &v, int x, int y) {
		u -= x; v -= y;
	});
}

template<typename _FlowT>
inline void xy2offset(Mat_<Vec<_FlowT, 2> > &uv)
{
	for_each_1c(DWHNC(uv), [](_FlowT *uv, int x, int y) {
		uv[0] -= x;
		uv[1] -= y;
	});
}


template<typename _FlowT>
inline void offset2xy(Mat_<_FlowT> &u, Mat_<_FlowT> &v)
{
	for_each_2c(DWHN1(u), DN1(v), [](_FlowT &u, _FlowT &v, int x, int y) {
		u += x; v += y;
	});
}

template<typename _FlowT>
inline void offset2xy(Mat_<Vec<_FlowT, 2> > &uv)
{
	for_each_1c(DWHNC(uv), [](_FlowT *uv, int x, int y) {
		uv[0] += x;
		uv[1] += y;
	});
}

Mat2f readFlowFile(const std::string &_filename);

void writeFlowFile(const Mat2f &uv, const std::string &_filename);

void readFlowKitti(Mat2f &uv, Mat1b &mask, const std::string& gtname);

void writeFlowKitti(Mat2f &uv, const std::string& floname);

int readFlowX(const std::string &flowFile, Mat2f &flow, Mat1b &mask);

void  saveMatches(const std::vector<Match> &matches, const std::string &_filename);

void  readMatches(std::vector<Match> &matches, const std::string &_filename);

//void  saveMatchesEX(const std::vector<Match> &matches, const std::string &_filename);

//void  readMatchesEX(std::vector<Match> &matches, const std::string &_filename);

Mat1f getFlowError(const Mat2f &flow, const Mat2f &gt);

double getAEE(const Mat2f &gt, const Mat2f &flow, Mat1b mask=Mat1b());

void crossCheck(const Mat2f &fflow, const Mat2f &bflow, std::vector<Match> &vMatches, float dT, int ds=3);

struct FlowPair
{
	std::string src;
	std::string tar;
	std::string info;
};

void saveFlowList(const std::string &file, const std::vector<FlowPair> &pairs);

void readFlowList(const std::string &file, std::vector<FlowPair> &pairs);

//std::string win2linux(const std::string &path, const std::string &winRoot, const std::string &linuxRoot);

Mat1f readEdgeFile(const std::string &filename, cv::Size size);

void saveEdgeFille(const std::string &filename, const Mat1f &_edge);

