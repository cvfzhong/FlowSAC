
#include"flowsac.h"
#include"utils.h"
#include"BFC/stdf.h"
#include"BFC/err.h"
#include<time.h>
#include<unistd.h>

Mat2f callEpicFlow(const std::string &srcfile, const std::string &tarfile, const std::string &sedEdgeFile, const std::string &matchFile, const std::string &datasetName);

Mat2f callRicFlow(const std::string &srcfile, const std::string &tarfile, const std::string &matchFile, const std::string &sedModelFile);

//struct Result
//{
//	double v[6];
//};

typedef cv::Vec6f Result;

void printResult(const std::vector<Result> &vr, FILE *fp)
{
	static const char *vname[] = { "Matches","Outlier","EpicFlow","RicFlow","FlowSAC","FlowSAC*" };
	for (int i = 0; i < Result::rows; ++i)
	{
		fprintf(fp, "%8s:", vname[i]);
		for (int j = 0; j < vr.size(); ++j)
		{
			fprintf(fp, "\t%3.2f", vr[j][i]);
		}
		fprintf(fp, "\n");
	}
}

void calcMatchError(const Mat2f &gt, std::vector<Match> &matches)
{
	for (auto &m : matches)
	{
		int x = int(m.x0 + 0.5), y = int(m.y0 + 0.5);
		if (uint(x) < gt.cols && uint(y) < gt.rows)
		{
			Vec2f dv = Vec2f(m.x1, m.y1) - gt(y, x);
			m.err = sqrt(dv.dot(dv));
		}
		else
			m.err = 1e6;
	}
}

void selMatchesWithCrossCheckT(const std::vector<Match> &matches, const std::vector<float> &crossCheckError, float errT, std::vector<Match> &sel)
{
	sel.clear();
	sel.reserve(matches.size());

	for (size_t i = 0; i < matches.size(); ++i)
		if (crossCheckError[i] < errT)
			sel.push_back(matches[i]);
}
float calcOutlierRatio(const std::vector<Match> &matches, float errT=3)
{
	int n = 0;
	for (auto &m : matches)
		if (m.err > errT)
			++n;
	return float(n) / (matches.size() + 1e-8f);
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		printf("Example: flowsac3 ../data/sintel-train/list-hard.txt flowfields1 sintel -o results.txt -L 3 -ucmT 0.1 -ransacT 2.6 -score SOD -r 10\n");
		return 0;
	}
	std::string listFile = argv[1], matchMethod = argv[2], datasetName = argv[3];
	const bool isKitti = stricmp(datasetName.c_str(), "kitti") == 0;

	std::string binDir = ff::GetDirectory(argv[0]);
	std::string sedModelFile = binDir + "./model.yml.gz";

	try
	{
		ff::ArgSet args;
		args.setArgs(argc - 4, argv + 4, false);

		ff::ArgSet flowsacArgs, fastArgs;
		flowsacArgs.setArgs("-fast -"); //force not fast
		flowsacArgs.setNext(&args);

		fastArgs.setArgs("-fast"); //force fast
		fastArgs.setNext(&args);

		std::string  dpath = ff::GetDirectory(listFile);

		std::vector<FlowPair> pairs;
		readFlowList(listFile, pairs);

		const float vT[] = { 3, 5, 8, 11, 15, 20, 25, 30, 40, 50 };
		const int nT = sizeof(vT) / sizeof(vT[0]);

		std::vector<std::vector<Result> > vResults(pairs.size());
		std::string jmatchFile = binDir + ff::StrFormat("./temp/%d.match", (int)getpid());

		for (size_t k = 0; k < pairs.size(); ++k)
		{
			printf("%d/%d : %s\n", k + 1, (int)pairs.size(), pairs[k].src.c_str());
			std::string srcFile = dpath + "images/" + pairs[k].src;
			std::string tarFile = dpath + "images/" + pairs[k].tar;

			std::string srcPath = pairs[k].src.substr(0, pairs[k].src.find_last_of('.'));
			std::string flowDir = dpath + "flow/" + srcPath + "/";

			std::string ucmFile = flowDir + "ucm.png";
			std::string edgeFile = flowDir + "sed.edge";

			std::string matchDir = dpath + "matchx/" + srcPath + "/";
			std::string matchxFile = matchDir + matchMethod + ".matchx";

			std::vector<Match> matches;
			readMatches(matches, matchxFile);

			std::vector<float>  crossCheckError(matches.size());
			for (size_t j = 0; j < matches.size(); ++j)
				crossCheckError[j] = matches[j].err;

			std::string gtFile = dpath + "gt/" + srcPath + ".flo";
			Mat2f gt=readFlowFile(gtFile);

			calcMatchError(gt, matches);

			vResults[k].resize(nT);


			for (int j = 0; j < nT; ++j)
			{
				std::vector<Match> jmatches;
				selMatchesWithCrossCheckT(matches, crossCheckError, vT[j], jmatches);

				Result r;

				r[0] = jmatches.size()/1000.0f;
				r[1] = calcOutlierRatio(jmatches)*100;

				saveMatches(jmatches, jmatchFile);
#if 1
				printf("%d ", j+1);
				fflush(stdout);
				Mat2f flow= callEpicFlow(srcFile, tarFile, edgeFile, jmatchFile, datasetName);
				r[2] = getAEE(gt, flow);

			//	printf("ricflow\n ");
				flow= callRicFlow(srcFile, tarFile, jmatchFile, sedModelFile);
				r[3] = getAEE(gt, flow);
#else
				Mat2f flow;
				r[2] = r[3] = 0;
#endif
			//	printf("flowsac\n ");
				{
					CFlowSAC flowSAC;
					flow = flowSAC(srcFile, tarFile, jmatchFile, ucmFile, edgeFile, datasetName, flowsacArgs);
					r[4] = getAEE(gt, flow);
				}
			//	printf("flowsac*\n");
				{
					CFlowSAC flowSAC;
					flow = flowSAC(srcFile, tarFile, jmatchFile, ucmFile, edgeFile, datasetName, fastArgs);
					r[5] = getAEE(gt, flow);
				}
				vResults[k][j] = r;
			}
			printf("\n");
			printResult(vResults[k], stdout);
		}

		std::vector<Result> mean(vResults.front());
		for (size_t i = 1; i < vResults.size(); ++i)
			for (size_t j = 0; j < mean.size(); ++j)
				mean[j] += vResults[i][j];

		for (auto &v : mean)
			v *= 1.0f/vResults.size();

		printf("mean\n");
		printResult(mean, stdout);

		std::string file = args.getd<std::string>("o", "./results3-"+matchMethod+".txt");
		FILE *fp = fopen(file.c_str(), "w");
		if (!fp)
		{
			printf("error:failed to open file %s\n", file.c_str());
			return -1;
		}

		for (size_t i = 0; i < vResults.size(); ++i)
		{
			fprintf(fp,"%04d : %s\n", i + 1, pairs[i].src.c_str());
			printResult(vResults[i], fp);
		}
		fprintf(fp, "mean:\n");
		printResult(mean, fp);
		fclose(fp);
		system(("rm \"" + jmatchFile + "\"").c_str());
	}
	catch (const std::exception &ec) {
		printf("error:%s\n", ec.what());
	}
	catch (...)
	{
	}

	return 0;
}

