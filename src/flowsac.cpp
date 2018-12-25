#include"BFC/stdf.h"
#include"BFC/err.h"
#include"BFC/portable.h"
#include"CVX/vis.h"
#include"opencv2/calib3d/calib3d.hpp"
#include"opencv2/highgui/highgui.hpp"
using namespace cv;

#include"scores.h"
#include"utils.h"
#include<time.h>

#include"flowsac.h"


typedef Match MatchX;

struct Group
{
	Mat1d  H;
	std::vector<MatchX>  vMatches;
	int state;
	float  score;
};

struct RegionX
{
	std::vector<MatchX>  vMatches;
	std::vector<Group>   vGroups;
	int state;
	int size = 0;
};


void findGroups(Size imgSize, std::vector<RegionX> &vRegions, float ransacReprojThreshold)
{
	Mat inlierMask;

	for (int i = 0; i < (int)vRegions.size(); ++i)
	{
		auto &r(vRegions[i]);
		r.vGroups.reserve(50);

		std::vector<Point2f> srcf, tarf;
		std::vector<MatchX> matches = r.vMatches;

		while(true)
		{
			if (matches.size() < 4)
				break;

			srcf.resize(0);
			tarf.resize(0);
			for (auto &m : matches)
			{
				srcf.push_back(Point2f(m.x0, m.y0));
				tarf.push_back(Point2f(m.x1, m.y1));
			}

			Mat1d H = cv::findHomography(srcf, tarf, inlierMask, RANSAC, ransacReprojThreshold);

			if (H.empty())
				break;
			else
			{
				std::vector<MatchX> inliers, outliers;

				const uchar *m = inlierMask.data;
				for (size_t k = 0; k < srcf.size(); ++k)
				{
					(m[k] ? inliers : outliers).push_back(matches[k]);
				}
				matches.swap(outliers);

				if (inliers.size() < 16)
					break;
				else
				{
					r.vGroups.push_back(Group());
					auto &g(r.vGroups.back());
					g.H = H;
					g.vMatches.swap(inliers);
				}
			}
		}
	}
}

void calcGroupScore(std::vector<RegionX> &vRegions, double beta = 0.1)
{
	for (auto &r : vRegions)
	{
		for (auto &g : r.vGroups)
		{
			double score = 0;
			for (auto &m : r.vMatches)
			{
				Point2f dv = transH(m.x0, m.y0, (double*)g.H.data) - Point2f(m.x1, m.y1);
				float d = dv.dot(dv);
				score += exp(-d*beta);
			}
			g.score = score;
		}
	}
}


void addMatches(const std::vector<MatchX> &src, std::vector<Match> &dest)
{
	for (auto &p : src)
		dest.push_back(p);
}

//collect matches with a downsampling rate #ds
void addMatches(const std::vector<MatchX> &src, std::vector<Match> &dest, double ds)
{
	if (ds < 1)
		ds = 1;

	double imax = src.size()-0.5;
	for (double i = 0; i < imax; i += ds)
	{
		dest.push_back(src[int(i+0.5)]);
	}
}

Mat2f _FlowSAC(const Mat3b &src, const Mat3b &tar, const Mat1i &seg, const std::vector<MatchX> &matches, std::vector<MatchX> &matchesFiltered, Interpolator &interp, FlowScore &score, int L, float ransacT, bool fast, double R)
{
	int nRegions = maxElem(seg) + 1;
	std::vector<RegionX> vRegions(nRegions);
	//collect matches of regions
	for (auto &m : matches)
	{
		int x = int(m.x0 + 0.5), y = int(m.y0 + 0.5);
		if (uint(x) < seg.cols && uint(y) < seg.rows)
		{
			int r = seg(y, x);
			vRegions[r].vMatches.push_back(MatchX(m));
		}
	}
	//compute size of regions
	for_each_1(DWHN1(seg), [&vRegions](int ri) {
		vRegions[ri].size++;
	});

	//the first region is always the region boundary, clear matches to prevent further processing
	vRegions.front().vMatches.clear(); 

	//grouping
	findGroups(src.size(), vRegions, ransacT);

	//ranking
	calcGroupScore(vRegions);

	std::vector<Match> matchx, matchNoDownsample;
	matchx.reserve(matches.size());
	matchNoDownsample.reserve(matches.size());

	//initialize states and collect the initial inlier set
	for (auto &r : vRegions)
	{
		for (auto &g : r.vGroups)
			g.state = 0;

		if (!r.vGroups.empty())
		{
			auto &g = r.vGroups.front();
			if (g.vMatches.size() > 30)
			{
				if(fast&&L>0) //if fast progressive estimation is required
				{
					//the downsampling rate
					double ds = R*R / (double(r.size) / g.vMatches.size());

					addMatches(g.vMatches, matchx, ds);
					addMatches(g.vMatches, matchNoDownsample);
				}
				else
					addMatches(g.vMatches, matchx);

				g.state = 1;
			}
		}
		r.state = 0;
	}
	//time_t beg = clock();

	//compute flow of the initial inlier set
	Mat2f flow = interp.exec(matchx, fast&&L>0);
	//printf("init. %d/%d, time=%.2lf\n", (int)matchx.size(), matchi.size(), double(clock()-beg)/CLOCKS_PER_SEC);

	if (L > 0)
	{
		int nmInit = matchx.size();

		//initialize the regional scores
		std::vector<double> regionScores;
		score.exec(flow, regionScores);

		std::vector<int>  curGroup(vRegions.size());

		for (int itr = 0; itr < L; ++itr)
		{
			size_t nm0 = matchx.size();

			for (size_t ri = 0; ri<vRegions.size(); ++ri)
			{
				auto &r(vRegions[ri]);
				int gs = -1;
				float v = -1;
				//search the next group with the largest ranking score
				for (size_t i = 0; i<r.vGroups.size(); ++i)
				{
					auto &g(r.vGroups[i]);
					if (g.state == 0 && g.score>v)
					{
						gs = (int)i;
						v = g.score;
					}
				}
				curGroup[ri] = gs;
				if (gs >= 0)
					addMatches(r.vGroups[gs].vMatches, matchx);
			}
			//compute the updated flow and regional scores
			flow = interp.exec(matchx, fast);

			std::vector<double> scorex;
			score.exec(flow, scorex);

			//roll back the inlier set
			matchx.resize(nm0);
			//re-add the groups masked as inliers to the inlier set
			for (size_t i = 0; i < vRegions.size(); ++i)
			{
				auto &r(vRegions[i]);
				int g = curGroup[i];
				if (g >= 0)
				{
					//r.state<0 means outlier group has been found, then the successive groups are all taken as outliers
					if (r.state >= 0 && scorex[i] > regionScores[i])
					{
						regionScores[i] = scorex[i];
						addMatches(r.vGroups[g].vMatches, matchx);
						r.vGroups[g].state = 1;
					}
					else
					{
						r.state = -1;
						r.vGroups[g].state = -1;
					}
				}
			}
#if 0
			//recompute flow and scores with the updated inlier set
			//slightly improve the accuracy but will double the computations
			flow = interp.exec(matchx);
			score.exec(flow, regionScores);
#endif
		}

		if (fast)
		{//the initial inlier set in #matchx is donwsampled, replace it with the set not downsampled for the final flow computation
			matchNoDownsample.insert(matchNoDownsample.end(), matchx.begin() + nmInit, matchx.end());
			matchx.swap(matchNoDownsample);
		}

		//the final flow, variational refinement is applied always
		flow = interp.exec(matchx, false);
	}

	matchesFiltered.swap(matchx);

	return flow;
}



Mat2f FlowSAC(const Mat3b &src, const Mat3b &tar, const Mat1i &seg, const std::vector<MatchX> &matches, std::vector<Match> &matchesFiltered, Interpolator &interp, FlowScore &score, int L, float ransacT, bool fast, double R)
{
	return _FlowSAC(src, tar, seg, matches, matchesFiltered, interp, score, L, ransacT, fast, R);
}



Mat2f CFlowSAC::operator()(const std::string &srcFile, const std::string &tarFile, const std::string &matchFile, const std::string &ucmFile, const std::string &edgeFile, const std::string &datasetName, ff::ArgSet &args)
{
	//parse optinal arguments
	const bool isKitti = stricmp(datasetName.c_str(), "kitti") == 0;
	const int L = args.getd<int>("L", 3);
	double ucmT = args.getd<float>("ucmT", isKitti ? 0.5 : 0.1);
	double ransacT = args.getd<float>("ransacT", isKitti ? 5.0 : 2.6);
	std::string scoreMethod = args.getd<std::string>("score", "SOD");
	bool fastMode = args.getd<bool>("fast", false);
	double R = args.getd<float>("r", 10);

	Mat3b src = imread(srcFile), tar = imread(tarFile);
	Mat1f sedEdge = readEdgeFile(edgeFile, src.size());

	std::vector<Match> matches;
	readMatches(matches, matchFile);

	Mat1f ucm = readFromPng(ucmFile, CV_32FC1);
	Mat1i region = segmentImage(ucm, matches, 0, 30, ucmT);

	std::unique_ptr<Interpolator> interp(createEpicInterpolator(src, srcFile, tar, tarFile, sedEdge, datasetName));
	

	std::unique_ptr<FlowScore> score(
		scoreMethod == "SOD" ? (FlowScore*)new SOD(src, tar, region, isKitti ? 1.0 : 0) : //for kitti dataset, smooth the images first
		scoreMethod == "SAD" ? (FlowScore*)new SAD(src, tar, region) :
		scoreMethod == "SNCC" ? (FlowScore*)new SNCC(src, tar, region, 7) :
		scoreMethod == "SCT" ? (FlowScore*)new SCT(src, tar, region, 7) :
		NULL
	);

	if (!score)
		FF_EXCEPTION("","unknown score method\n");

	Mat2f flow = FlowSAC(src, tar, region, matches, matchesFiltered, *interp, *score, L, ransacT, fastMode, R);

	return flow;
}

