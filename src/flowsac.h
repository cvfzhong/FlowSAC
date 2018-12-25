#pragma once

#include"CVX/core.h"
#include"BFC/argv.h"
using namespace cv;

class Match
{
public:
	float x0, y0;
	float x1, y1;

public:
	Match()
	{}
	Match(float _x0, float _y0, float _x1, float _y1)
		:x0(_x0), y0(_y0), x1(_x1), y1(_y1)
	{}
};


Mat1i segmentImage(const Mat1f &ucm, const std::vector<Match> &vmatch, int minRegionSize, int minRegionMatches, float ucmThreshold);


class FlowScore
{
public:
	virtual void exec(const Mat2f &flow, std::vector<double> &regionScores) = 0;

	virtual ~FlowScore() {}
};

class Interpolator
{
public:
	virtual Mat2f exec(const std::vector<Match> &vMatches, bool fast) = 0;

	virtual ~Interpolator() {}
};

Mat2f FlowSAC(const Mat3b &src, const Mat3b &tar, const Mat1i &seg, const std::vector<Match> &matches, std::vector<Match> &matchesFiltered, Interpolator &interp, FlowScore &score, int nMaxGroups, float ransacT, bool fast, double R);

Interpolator* createEpicInterpolator(const Mat3b &src, const std::string &srcFile, const Mat3b &tar, const std::string &tarFile, const Mat1f &sedEdge, const std::string &datasetName);


class CFlowSAC
{
public:
	std::vector<Match> matchesFiltered;
public:
	Mat2f operator()(const std::string &srcFile, const std::string &tarFile, const std::string &matchFile, const std::string &ucmFile, const std::string &edgeFile, const std::string &datasetName, ff::ArgSet &args);
};

