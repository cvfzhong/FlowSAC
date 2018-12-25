
#include"flowsac.h"
#include"utils.h"
#include"BFC/stdf.h"
#include"BFC/err.h"
#include<time.h>

//Mat2f callEpicFlow(const std::string &epicExeFile, const std::string &srcfile, const std::string &tarfile, const std::string &sedEdgeFile, const std::string &matchFile, const std::string &flowFile, const std::string &datasetName)
//{
//	char cmd[1024];
//	sprintf(cmd, "%s %s %s %s %s %s -%s", epicExeFile.data(), srcfile.data(), tarfile.data(), sedEdgeFile.data(), matchFile.data(), flowFile.data(), datasetName.c_str());
//	system(cmd);
//	return readFlowFile(flowFile);
//}

//Mat2f callRicFlow(const std::string &ricExeFile, const std::string &srcfile, const std::string &tarfile, const std::string &matchFile, const std::string &flowFile)
//{
//	char cmd[1024];
//	sprintf(cmd, "%s %s %s %s %s", ricExeFile.data(), srcfile.data(), tarfile.data(), matchFile.data(), flowFile.data());
//	//printf("%s\n", cmd);
//	system(cmd);
//	return readFlowFile(flowFile);
//}

Mat2f callEpicFlow(const std::string &srcfile, const std::string &tarfile, const std::string &sedEdgeFile, const std::string &matchFile, const std::string &datasetName);

Mat2f callRicFlow(const std::string &srcfile, const std::string &tarfile, const std::string &matchFile, const std::string &sedModelFile);

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		printf("Example: batch ../data/sintel-train/list-all.txt flowfields sintel -method [flowsac|epicflow|ricflow] -o results.txt -L 3 -ucmT 0.1 -ransacT 2.6 -score SOD -fast -r 10\n");
		return 0;
	}
	std::string listFile = argv[1], matchMethod=argv[2], datasetName = argv[3];
	const bool isKitti = stricmp(datasetName.c_str(), "kitti") == 0;

	std::string binDir = ff::GetDirectory(argv[0]);
	std::string sedModelFile = binDir + "./model.yml.gz";

	try
	{
		ff::ArgSet args;
		args.setArgs(argc - 4, argv + 4, false);

		std::string method = args.getd<std::string>("method", "flowsac");
		ff::str2lower(method);

		std::string  dpath = ff::GetDirectory(listFile);

		std::vector<FlowPair> pairs;
		readFlowList(listFile, pairs);
		
		std::vector<std::vector<double> > vResults(pairs.size());

		for (size_t i = 0; i < pairs.size(); ++i)
		{
			std::string srcFile = dpath + "images/" + pairs[i].src;
			std::string tarFile = dpath + "images/" + pairs[i].tar;
			
			std::string srcPath=pairs[i].src.substr(0, pairs[i].src.find_last_of('.'));
			std::string flowDir = dpath + "flow/" + srcPath + "/";
			
			std::string matchFile = flowDir + matchMethod+".match";
			std::string ucmFile = flowDir + "ucm.png";
			std::string edgeFile = flowDir + "sed.edge";

			int64 beg = cv::getTickCount();
			Mat2f flow;

			if (method == "flowsac")
			{
				CFlowSAC flowSAC;
				flow = flowSAC(srcFile, tarFile, matchFile, ucmFile, edgeFile, datasetName, args);
			}
			else if (method == "epicflow")
				flow = callEpicFlow(srcFile, tarFile, edgeFile, matchFile, datasetName);
			else if (method == "ricflow")
				flow = callRicFlow(srcFile, tarFile, matchFile, sedModelFile);
			else
				FF_EXCEPTION("", "unknown method");

			float timeCost = float(cv::getTickCount() - beg) / cv::getTickFrequency();

			std::string gtFile = dpath + "gt/" + srcPath + (isKitti? ".png" : ".flo");
			Mat1b gtMask;
			Mat2f gt;
			//read .flo or .png flow file according to the file extension
			readFlowX(gtFile, gt, gtMask);
			double aee = getAEE(gt, flow, gtMask);

			vResults[i].push_back(aee);
			vResults[i].push_back(timeCost);

			printf("%04d: aee=%.2lf,time=%.2lf\t%s\n", (int)i, aee, timeCost, pairs[i].src.c_str());
		}

		std::vector<double> mean(vResults.front());
		for (size_t i = 1; i < vResults.size(); ++i)
			for (size_t j = 0; j < mean.size(); ++j)
				mean[j] += vResults[i][j];

		for (auto &v : mean)
			v /= vResults.size();

		printf("mean: aee=%.2lf,time=%.2lf\n", mean[0], mean[1]);

		std::string file = args.getd<std::string>("o", "./results.txt");
		FILE *fp = fopen(file.c_str(), "w");
		for (size_t i = 0; i < vResults.size(); ++i)
		{
			fprintf(fp,"%04d: aee=%.2lf,time=%.2lf\t%s\n", (int)i, vResults[i][0], vResults[i][1], pairs[i].src.c_str());
		}
		fprintf(fp,"mean: aee=%.2lf,time=%.2lf\n", mean[0], mean[1]);
		fclose(fp);
	}
	catch (const std::exception &ec) {
		printf("error:%s\n", ec.what());
	}
	catch (...)
	{
	}

	return 0;
}

